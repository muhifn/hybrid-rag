"""Switchable structured-data layer: text-to-SQL (SQLite) or text-to-Cypher (Neo4j)."""

from app.graph.base import GraphResult, GraphStore
from app.graph.tables import TableSpec, load_tables

__all__ = ["GraphResult", "GraphStore", "TableSpec", "load_tables", "build_graph_store"]


def build_graph_store(settings) -> GraphStore:
    """Open the configured backend. The import is lazy so neo4j stays optional."""
    if settings.graph_backend == "sqlite":
        from app.graph.sqlite_store import SQLiteGraphStore

        return SQLiteGraphStore(settings)
    if settings.graph_backend == "neo4j":
        from app.graph.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(settings)
    raise ValueError(f"Unknown GRAPH_BACKEND {settings.graph_backend!r}")


def ingest_tables(settings, tables: list[TableSpec]) -> None:
    """(Re)build the configured backend from the CSV tables."""
    if settings.graph_backend == "sqlite":
        from app.graph.sqlite_store import build_sqlite_database

        build_sqlite_database(settings.sqlite_path, tables)
    elif settings.graph_backend == "neo4j":
        from app.graph.neo4j_store import ingest_into_neo4j

        ingest_into_neo4j(settings, tables)
    else:
        raise ValueError(f"Unknown GRAPH_BACKEND {settings.graph_backend!r}")
