"""Text-to-Cypher backend: Neo4j accessed through LangChain's `Neo4jGraph`.

Table -> graph mapping:
  * one table       -> one node label       (`pegawai` -> `:Pegawai`)
  * one row         -> one node; every column becomes a property
  * one foreign key -> one directed relationship named after the column
                       without `_id`         (`unit_id` -> `-[:UNIT]->`)

That makes multi-hop questions natural, e.g.
  (:Proyek)-[:UNIT]->(:Unit)-[:KEPALA_PEGAWAI]->(:Pegawai)
which needs two chained JOINs in SQL.

REQUIRES the APOC plugin: `Neo4jGraph` builds its schema with
`apoc.meta.data()`. The bundled docker-compose.yml enables it.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_neo4j import Neo4jGraph

from app.config import Settings
from app.graph.tables import TableSpec, coerce
from app.text_utils import tokenize

log = logging.getLogger(__name__)

FORBIDDEN = {
    "create", "merge", "delete", "detach", "set", "remove", "drop", "load",
    "foreach", "periodic", "constraint", "index",
}
ALLOWED_START = ("match", "optional", "with", "unwind", "return", "call")


def node_label(table: str) -> str:
    """pegawai -> Pegawai, unit_kerja -> UnitKerja"""
    return "".join(part.capitalize() for part in table.split("_") if part)


def relationship_type(column: str) -> str:
    """unit_id -> UNIT, penanggung_jawab_pegawai_id -> PENANGGUNG_JAWAB_PEGAWAI"""
    base = column[:-3] if column.endswith("_id") else column
    return base.upper() or "RELATED_TO"


def _connect(settings: Settings, refresh_schema: bool) -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        refresh_schema=refresh_schema,
        enhanced_schema=refresh_schema,  # adds distinct values / ranges per property
    )


def ingest_into_neo4j(settings: Settings, tables: list[TableSpec]) -> None:
    """Replace the graph contents with the given tables."""
    graph = _connect(settings, refresh_schema=False)
    by_name = {t.name: t for t in tables}
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        for t in tables:
            label = node_label(t.name)
            graph.query(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.`{t.pk}` IS UNIQUE")
            rows = [{c: coerce(r.get(c), t.types[c]) for c in t.columns} for r in t.rows]
            graph.query(f"UNWIND $rows AS row CREATE (n:{label}) SET n = row", params={"rows": rows})

        relationships = 0
        for t in tables:
            for col, ref in t.fks.items():
                if ref not in by_name:
                    continue
                result = graph.query(
                    f"MATCH (a:{node_label(t.name)}), (b:{node_label(ref)}) "
                    f"WHERE a.`{col}` IS NOT NULL AND a.`{col}` = b.`{by_name[ref].pk}` "
                    f"CREATE (a)-[:{relationship_type(col)}]->(b) RETURN count(*) AS created"
                )
                relationships += result[0]["created"] if result else 0
    finally:
        graph.close()
    log.info("Loaded Neo4j: %d nodes, %d relationships",
             sum(len(t.rows) for t in tables), relationships)


class Neo4jGraphStore:
    language = "cypher"

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            self.graph = _connect(settings, refresh_schema=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cannot open Neo4j at {settings.neo4j_uri}: {type(exc).__name__}: {exc}\n"
                "Start it with `docker compose up -d neo4j` (APOC must be enabled), "
                "or set GRAPH_BACKEND=sqlite."
            ) from exc

    def table_names(self) -> list[str]:
        return sorted(self.graph.structured_schema.get("node_props", {}))

    def vocabulary(self) -> set[str]:
        words: set[str] = set()
        for label, props in self.graph.structured_schema.get("node_props", {}).items():
            words.update(tokenize(label))
            for p in props:
                words.update(t for t in tokenize(p["property"]) if t != "id")
        return words

    def schema_description(self) -> str:
        return "Neo4j GRAPH SCHEMA\n\n" + self.graph.schema

    def validate(self, query: str) -> str | None:
        q = (query or "").strip().rstrip(";").strip()
        if not q:
            return "empty query"
        low = q.lower()
        if not low.startswith(ALLOWED_START):
            return "query must start with MATCH / OPTIONAL MATCH / WITH / UNWIND / RETURN / CALL"
        if ";" in q:
            return "multiple statements are not allowed"
        tokens = set(tokenize(low))
        bad = sorted(tokens & FORBIDDEN)
        if bad:
            return "contains write clauses: " + ", ".join(bad)
        if "return" not in tokens:
            return "query must have a RETURN clause"
        return None

    def run(self, query: str, limit: int = 50) -> tuple[list[str], list[dict[str, Any]]]:
        q = query.strip().rstrip(";").strip()
        if "limit" not in tokenize(q.lower()):
            q = f"{q} LIMIT {limit}"
        # READ access mode: the server itself rejects writes, independent of validate().
        rows = self.graph.query(q, session_params={"default_access_mode": "READ"})
        return (list(rows[0]) if rows else []), rows

    def examples(self) -> list[tuple[str, str]]:
        schema = self.graph.structured_schema
        props = schema.get("node_props", {})
        rels = schema.get("relationships", [])

        def numeric(label: str) -> list[str]:
            return [p["property"] for p in props.get(label, [])
                    if p.get("type") in ("INTEGER", "FLOAT") and not p["property"].endswith("_id")]

        def text(label: str) -> str:
            names = [p["property"] for p in props.get(label, [])
                     if p.get("type") == "STRING" and not p["property"].endswith("_id")]
            return names[0] if names else "id"

        out: list[tuple[str, str]] = []
        for rel in rels:
            nums = numeric(rel["start"])
            if nums:
                out.append((
                    f"What is the total {nums[0]} per {rel['end']}?",
                    f"MATCH (a:{rel['start']})-[:{rel['type']}]->(b:{rel['end']}) "
                    f"RETURN b.`{text(rel['end'])}` AS {rel['end'].lower()}, sum(a.`{nums[0]}`) AS total "
                    f"ORDER BY total DESC",
                ))
                break
        for label in props:
            nums = numeric(label)
            if nums:
                out.append((
                    f"List the 5 {label} nodes with the highest {nums[0]}.",
                    f"MATCH (n:{label}) RETURN n.`{text(label)}` AS name, n.`{nums[0]}` AS value "
                    f"ORDER BY value DESC LIMIT 5",
                ))
                break
        return out

    def close(self) -> None:
        self.graph.close()
