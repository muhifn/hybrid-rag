"""Ingestion: documents -> chunks -> Qdrant, and CSV tables -> graph backend."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

from app.config import Settings
from app.graph import ingest_tables, load_tables
from app.ingestion import load_chunk_jsonl, load_documents, split_documents
from app.vectorstore import VectorIndex

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents: int
    chunks: int
    tables: int
    table_rows: int
    graph_backend: str
    seconds: float
    source: str = "documents"  # "ocr_chunks" | "documents"

    def as_dict(self) -> dict:
        return asdict(self)


def _load_chunks(settings: Settings) -> tuple[list, int, str]:
    """Chunks for the vector store, preferring the OCR pipeline's output.

    Returns (chunks, document_count, source_label). OCR chunks are used as-is:
    they were split on page and table boundaries, which the character splitter
    cannot reproduce. Plain documents are split here as before.
    """
    if settings.use_ocr_chunks:
        chunks = load_chunk_jsonl(settings.ocr_chunks_file)
        if chunks:
            documents = len({c.metadata.get("source") for c in chunks})
            log.info("Using OCR chunks from %s", settings.ocr_chunks_file)
            return chunks, documents, "ocr_chunks"
        log.info("No OCR chunks at %s; loading %s instead",
                 settings.ocr_chunks_file, settings.documents_dir)

    docs = load_documents(settings.documents_dir)
    return split_documents(docs, settings.chunk_size, settings.chunk_overlap), len(docs), "documents"


def run_ingestion(settings: Settings, index: VectorIndex, recreate: bool = True) -> IngestStats:
    """Rebuild both stores from `data/`.

    Callers must close any open graph store first: on Windows, SQLite's file
    cannot be replaced while a connection holds it.
    """
    t0 = time.perf_counter()

    chunks, document_count, source = _load_chunks(settings)
    index.ensure_collection(recreate=recreate)
    if chunks:
        index.add(chunks)

    tables = load_tables(settings.tables_dir, settings.fk_overrides)
    if tables:
        ingest_tables(settings, tables)
    else:
        log.info("No tables to ingest — graph backend will not be available")

    stats = IngestStats(
        documents=document_count, chunks=len(chunks), tables=len(tables),
        table_rows=sum(len(t.rows) for t in tables), graph_backend=settings.graph_backend,
        seconds=round(time.perf_counter() - t0, 2), source=source,
    )
    log.info("Ingestion finished: %s", stats.as_dict())
    return stats
