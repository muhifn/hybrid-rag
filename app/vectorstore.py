"""Qdrant vector store plus the dense / BM25 / hybrid retrievers built on it."""

from __future__ import annotations

import gc
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.text_utils import tokenize

log = logging.getLogger(__name__)

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")

# Program names as questions phrase them -> the program their documents answer
# from. The corpus holds several study programs' panduan side by side; a
# question that names one must not be answered from the others' near-identical
# visi/misi/kurikulum sections.
_PROGRAM_ALIASES: dict[str, tuple[str, ...]] = {
    "teknik informatika": ("teknik informatika", "informatika", "ti"),
    "sistem dan teknologi informasi": ("sistem dan teknologi informasi", "teknologi informasi", "sti"),
    "teknik elektro": ("teknik elektro", "elektro", "te"),
    "teknik mesin": ("teknik mesin", "tm"),
    "mekatronika": ("mekatronika",),
}
_PROGRAM_PATTERNS = {
    program: tuple(re.compile(rf"\b{re.escape(alias)}\b") for alias in aliases)
    for program, aliases in _PROGRAM_ALIASES.items()
}


class VectorIndex:
    """Owns the Qdrant client and the LangChain `QdrantVectorStore` on top of it.

    The same code runs against three deployments; only the client differs:
      embedded -> QdrantClient(path=...)              local folder, zero install
      server   -> QdrantClient(url="http://...:6333") Docker / self-hosted
      server   -> QdrantClient(url=CLOUD, api_key=..) Qdrant Cloud free tier

    Embedded mode locks its folder: only ONE process may open it. Do not run
    `scripts/ingest.py` while the API is up in embedded mode - use the
    `/admin/ingest` endpoint instead, or switch to server mode.
    """

    def __init__(self, settings: Settings, embeddings: Embeddings, dimension: int):
        self.settings = settings
        self.embeddings = embeddings
        self.dimension = dimension
        self.collection = settings.qdrant_collection
        if settings.qdrant_mode == "embedded":
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
        else:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            embedding=embeddings,
            validate_collection_config=False,  # collection may not exist yet
        )

    # ------------------------------------------------------------ collection
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.exists():
            self._reset_collection()
            return
        if not self.exists():
            self._create_collection()
            return
        existing = self._existing_dimension()
        if existing != self.dimension:
            raise RuntimeError(
                f"Collection '{self.collection}' has dim={existing} but the embedding model "
                f"produces dim={self.dimension}. Re-ingest with --recreate."
            )

    def _existing_dimension(self) -> int:
        return self.client.get_collection(self.collection).config.params.vectors.size

    def _create_collection(self) -> None:
        self.client.create_collection(
            self.collection,
            # Cosine: text similarity is about vector direction, not length.
            vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
        )
        log.info("Created collection %s (dim=%d, cosine)", self.collection, self.dimension)

    def _reset_collection(self) -> None:
        """Empty the collection, recreating it only if the vector size changed.

        Dropping and recreating is avoided on purpose: in embedded mode on
        Windows, qdrant-client cannot delete the collection's storage.sqlite
        while its (orphaned) connection is open, and `create_collection` then
        reloads the OLD points from that file - stale chunks would silently
        survive re-ingestion.
        """
        if self._existing_dimension() == self.dimension:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(filter=models.Filter()),  # match all
                wait=True,
            )
            log.info("Cleared all points from collection %s", self.collection)
        else:
            self.client.delete_collection(self.collection)
            if self.settings.qdrant_mode == "embedded":
                gc.collect()  # finalise the orphaned SQLite connection
                leftover = self.settings.qdrant_path / "collection" / self.collection
                if leftover.exists():
                    shutil.rmtree(leftover)
            self._create_collection()

        remaining = self.client.count(self.collection, exact=True).count
        if remaining:
            raise RuntimeError(f"Collection {self.collection} still holds {remaining} points after reset")

    # ------------------------------------------------------------- indexing
    @staticmethod
    def point_id(chunk_id: str) -> str:
        # Deterministic UUIDv5: re-ingesting a chunk overwrites it instead of duplicating.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def add(self, chunks: list[Document], batch_size: int = 64) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.store.add_documents(batch, ids=[self.point_id(c.metadata["chunk_id"]) for c in batch])
            log.info("Indexed %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))
        return len(chunks)

    def all_documents(self) -> list[Document]:
        """Read every chunk back out of Qdrant.

        Qdrant is the single source of truth: the BM25 index is rebuilt from
        it at startup, so dense and sparse retrieval always see the same corpus.
        """
        if not self.exists():
            return []
        docs: list[Document] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                payload = p.payload or {}
                docs.append(Document(page_content=payload.get("page_content", ""),
                                     metadata=payload.get("metadata", {})))
            if offset is None:
                break
        docs.sort(key=lambda d: d.metadata.get("chunk_id", ""))
        return docs

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class RetrieverSet:
    """The three retrieval strategies compared in the evaluation."""

    dense: BaseRetriever
    bm25: BaseRetriever | None
    hybrid: BaseRetriever
    top_k: int
    titles: list[str] = field(default_factory=list)  # for the router prompt

    def get(self, mode: str) -> BaseRetriever:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {RETRIEVAL_MODES}")
        if mode == "bm25":
            if self.bm25 is None:
                raise RuntimeError("BM25 index is empty - ingest documents first")
            return self.bm25
        return self.dense if mode == "dense" else self.hybrid

    def retrieve(self, query: str, mode: str = "hybrid") -> list[Document]:
        # EnsembleRetriever returns the fused union of both pools; collapse
        # heading-only near-duplicates, lift table chunks for list queries,
        # then cut to top_k.
        docs = self.get(mode).invoke(query)
        docs = self._collapse_near_duplicates(docs)
        docs = self._lift_table_chunks_for_list(query, docs)
        return self._prefer_mentioned_program(query, docs)[: self.top_k]

    # Tokens that mean "the answer is a list of TABLE rows". Critically these
    # exclude generic "apa saja" ("Apa saja profil lulusan prodi X?" is answered
    # from prose, not a table).
    _TABLE_LIST_INTENT = ("daftar", "daftarkan", "daftarnya")

    @staticmethod
    def _lift_table_chunks_for_list(query: str, docs: list[Document]) -> list[Document]:
        """For table-list questions, lift at most 3 table chunks to the front.

        "daftar mata kuliah pengayaan yang dapat direkognisi dengan prestasi"
        ranked the real recognition tables #10/#14 behind prose intros, so the
        model had no table rows to enumerate. Lifting tables fixes that, but
        unrestricted lifting pushed off-topic tables into every "apa saja"
        answer (Q9 then invented a Teknik Informatika profile for a Sistem
        Informasi question) - so this is capped and keyed to explicit list
        words, never to the generic "apa saja".
        """
        q = query.lower()
        if not any(token in q for token in RetrieverSet._TABLE_LIST_INTENT):
            return docs

        # Only lift the best few tables; off-topic tables from the pool must not
        # crowd out the actually relevant prose.
        lifted: list[Document] = []
        rest: list[Document] = []
        for doc in docs:
            if doc.metadata.get("content_type") == "table" and len(lifted) < 3:
                lifted.append(doc)
            else:
                rest.append(doc)
        return lifted + rest


    @staticmethod
    def _collapse_near_duplicates(docs: list[Document]) -> list[Document]:
        """Drop chunks that are a strict content-subset of a richer chunk on the
        same page, keeping the richer one at the better of the two ranks.

        Table sections often ship a tiny heading-only chunk alongside the real
        table chunk (both carry the same "Bagian:" prefix). The tiny one wins
        BM25 on length, occupies a top-k slot, and the model then honestly says
        the promised table "is not available in the context". Example: p37 of
        the Pengayaan panduan - a 316-char heading chunk beat the 2213-char
        PKM recognition table it duplicates.
        """
        if len(docs) < 2:
            return docs
        norms = [re.sub(r"[^a-z0-9]+", "", (d.page_content or "").lower()) for d in docs]

        def key(doc: Document) -> tuple[str, object]:
            return (doc.metadata.get("source") or "", doc.metadata.get("page"))

        pages: dict[tuple[str, object], list[int]] = {}
        for i, doc in enumerate(docs):
            if norms[i]:
                pages.setdefault(key(doc), []).append(i)

        dropped: set[int] = set()
        # container idx -> best rank to inherit from the chunks it absorbs
        promoted: dict[int, int] = {}
        for idxs in pages.values():
            for i in idxs:
                container = None
                for j in idxs:
                    if j == i or j in dropped:
                        continue
                    # keep only if j strictly contains i's text (not equal peers)
                    if len(norms[j]) > len(norms[i]) * 1.05 and norms[i] in norms[j]:
                        if container is None or len(norms[j]) > len(norms[container]):
                            container = j
                if container is not None:
                    dropped.add(i)
                    promoted[container] = min(promoted.get(container, container), i)

        survivors = sorted(((promoted.get(i, i), i) for i in range(len(docs)) if i not in dropped))
        return [docs[i] for _, i in survivors]

    @staticmethod
    def _prefer_mentioned_program(query: str, docs: list[Document]) -> list[Document]:
        """Stable-boost documents of a program the query explicitly names.

        "visi misi prodi teknik informatika" must not be answered from the
        Teknik Elektro/Mesin panduan merely because their visi/misi sections
        read similarly. Matching titles move to the front; relevance order
        within each group is preserved.
        """
        q = query.lower()
        wanted = [program for program, patterns in _PROGRAM_PATTERNS.items()
                  if any(pattern.search(q) for pattern in patterns)]
        if not wanted:
            return docs

        def rank(doc: Document) -> int:
            title = (doc.metadata.get("title") or "").lower()
            return 0 if any(program in title for program in wanted) else 1

        return sorted(docs, key=rank)


def build_retrievers(index: VectorIndex, settings: Settings) -> RetrieverSet:
    """Dense (Qdrant) + sparse (BM25) fused with weighted Reciprocal Rank Fusion.

    Dense retrieval misses rare exact tokens - document codes, acronyms such as
    KAK/RAB, names, numbers. BM25 is strong exactly there. RRF combines the two
    by rank rather than raw score, because cosine scores (0..1) and BM25 scores
    (unbounded) are not comparable.
    """
    k, pool = settings.top_k, max(settings.retrieval_candidates, settings.top_k)
    dense = index.store.as_retriever(search_kwargs={"k": k})
    dense_pool = index.store.as_retriever(search_kwargs={"k": pool})

    corpus = index.all_documents()
    if not corpus:
        log.warning("Vector store is empty; BM25 disabled and hybrid falls back to dense")
        return RetrieverSet(dense=dense, bm25=None, hybrid=dense, top_k=k)

    bm25 = BM25Retriever.from_documents(corpus, k=k, preprocess_func=tokenize)
    bm25_pool = BM25Retriever.from_documents(corpus, k=pool, preprocess_func=tokenize)
    hybrid = EnsembleRetriever(
        retrievers=[dense_pool, bm25_pool],
        weights=[1.0 - settings.bm25_weight, settings.bm25_weight],
        c=settings.rrf_c,
        id_key="chunk_id",  # dedupe by chunk id, not by page_content
    )
    log.info("Retrievers ready over %d chunks (k=%d, pool=%d, bm25_weight=%.2f)",
             len(corpus), k, pool, settings.bm25_weight)
    titles = sorted({d.metadata.get("title", "") for d in corpus} - {""})
    return RetrieverSet(dense=dense, bm25=bm25, hybrid=hybrid, top_k=k, titles=titles)
