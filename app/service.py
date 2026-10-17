"""RAGService: builds and owns every component; shared by the API, CLI, and evaluation."""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterator

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings
from app.graph import GraphStore, build_graph_store
from app.graph.query_engine import GraphQueryEngine
from app.indexing import IngestStats, run_ingestion
from app.llm import build_chat_model, build_embeddings, embedding_dimension, list_server_models, resolve_model_name
from app.memory import ReadWriteLock, SessionStore
from app.pipeline import HybridRAGPipeline, RAGResult
from app.router import QueryRouter
from app.vectorstore import RetrieverSet, VectorIndex, build_retrievers

log = logging.getLogger(__name__)


class NotReadyError(RuntimeError):
    """Raised when chat is requested before any data has been ingested."""


class RAGService:
    def __init__(self, settings: Settings, llm: BaseChatModel | None = None,
                 embeddings: Embeddings | None = None):
        self.settings = settings
        self.embeddings = embeddings or build_embeddings(settings)
        self.dimension = embedding_dimension(self.embeddings)
        log.info("Embedding model ready: %s (dim=%d)", settings.embedding_model, self.dimension)

        if llm is None:
            self.model_name = resolve_model_name(settings)
            llm = build_chat_model(settings, self.model_name)
        else:
            self.model_name = getattr(llm, "model_name", type(llm).__name__)
        self.llm = llm

        self.index = VectorIndex(settings, self.embeddings, self.dimension)
        self.sessions = SessionStore(settings.max_sessions)
        self._rw = ReadWriteLock()
        self._vector_lock = threading.Lock() if settings.qdrant_mode == "embedded" else None

        self.retrievers: RetrieverSet | None = None
        self.graph_store: GraphStore | None = None
        self.graph_engine: GraphQueryEngine | None = None
        self.router: QueryRouter | None = None
        self.pipeline: HybridRAGPipeline | None = None
        self.graph_error: str | None = None
        self._load_components()

    # ----------------------------------------------------------- lifecycle
    def _load_components(self) -> None:
        """(Re)build retrievers, graph store, router, and pipeline from the stores."""
        self.retrievers = build_retrievers(self.index, self.settings)

        self.graph_store, self.graph_engine, self.graph_error = None, None, None
        try:
            self.graph_store = build_graph_store(self.settings)
            self.graph_engine = GraphQueryEngine(
                self.graph_store, self.llm,
                max_attempts=self.settings.graph_max_attempts, row_limit=self.settings.graph_row_limit,
            )
        except Exception as exc:  # noqa: BLE001 - run vector-only rather than not at all
            self.graph_error = f"{type(exc).__name__}: {exc}"
            log.warning("Graph backend unavailable, running vector-only: %s", self.graph_error)

        self.router = QueryRouter(
            self.llm,
            table_names=self.graph_store.table_names() if self.graph_store else [],
            schema_terms=self.graph_store.vocabulary() if self.graph_store else set(),
            document_titles=self.retrievers.titles,
            use_structured_output=self.settings.router_structured_output,
        )

        if self.index.count() == 0 and self.graph_store is None:
            self.pipeline = None
            log.warning("No data ingested yet. POST /admin/ingest or run `python -m scripts.ingest`.")
            return
        self.pipeline = HybridRAGPipeline(
            self.llm, self.router, self.retrievers, self.graph_engine, self.sessions,
            max_history_turns=self.settings.max_history_turns, vector_lock=self._vector_lock,
        )

    def ingest(self, recreate: bool = True) -> IngestStats:
        """Rebuild all stores. Blocks new chat requests until finished."""
        with self._rw.write():
            if self.graph_store is not None:
                self.graph_store.close()  # release the SQLite file before replacing it
                self.graph_store = None
            stats = run_ingestion(self.settings, self.index, recreate=recreate)
            self._load_components()
            return stats

    def close(self) -> None:
        if self.graph_store is not None:
            self.graph_store.close()
        self.index.close()

    # ---------------------------------------------------------------- chat
    def _require_pipeline(self) -> HybridRAGPipeline:
        if self.pipeline is None:
            raise NotReadyError("No data ingested yet. POST /admin/ingest or run `python -m scripts.ingest`.")
        return self.pipeline

    def ask(self, question: str, **kwargs: Any) -> RAGResult:
        with self._rw.read():
            return self._require_pipeline().ask(question, **kwargs)

    def stream(self, question: str, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        # Hold the read lock only while generating; a client that disconnects
        # mid-stream closes the generator, which releases it.
        with self._rw.read():
            yield from self._require_pipeline().stream(question, **kwargs)

    # -------------------------------------------------------------- status
    def health(self) -> dict[str, Any]:
        try:
            served = list_server_models(self.settings, timeout=3.0)
            llm_status = {"reachable": True, "model": self.model_name,
                          "model_loaded": self.model_name in served or not served}
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:200]
            if "401" in error:
                error += " -> LM Studio requires an API token: set LLM_API_KEY in .env"
            llm_status = {"reachable": False, "model": self.model_name, "error": error}
        return {
            "ready": self.pipeline is not None,
            "llm": llm_status,
            "embeddings": {"model": self.settings.embedding_model, "dimension": self.dimension,
                           "backend": self.settings.embedding_backend,
                           "device": (self.settings.embedding_device
                                      if self.settings.embedding_backend == "huggingface"
                                      else self.settings.llm_base_url)},
            "vector_store": {"mode": self.settings.qdrant_mode, "collection": self.settings.qdrant_collection,
                             "chunks": self.index.count()},
            "graph": {"backend": self.settings.graph_backend, "available": self.graph_store is not None,
                      "tables": self.graph_store.table_names() if self.graph_store else [],
                      "error": self.graph_error},
            "sessions": len(self.sessions),
        }
