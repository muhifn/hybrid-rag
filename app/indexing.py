"""Ingestion: documents -> chunks -> Qdrant, and CSV tables -> graph backend."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

from app.config import Settings
from app.graph import ingest_tables, load_tables
from app.ingestion import load_documents, split_documents
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

    def as_dict(self) -> dict:
        return asdict(self)


def run_ingestion(settings: Settings, index: VectorIndex, recreate: bool = True) -> IngestStats:
    """Rebuild both stores from `data/`.

    Callers must close any open graph store first: on Windows, SQLite's file
    cannot be replaced while a connection holds it.
    """
    t0 = time.perf_counter()

    docs = load_documents(settings.documents_dir)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    index.ensure_collection(recreate=recreate)
    if chunks:
        index.add(chunks)

    tables = load_tables(settings.tables_dir, settings.fk_overrides)
    ingest_tables(settings, tables)

    stats = IngestStats(
        documents=len(docs), chunks=len(chunks), tables=len(tables),
        table_rows=sum(len(t.rows) for t in tables), graph_backend=settings.graph_backend,
        seconds=round(time.perf_counter() - t0, 2),
    )
    log.info("Ingestion finished: %s", stats.as_dict())
    return stats
