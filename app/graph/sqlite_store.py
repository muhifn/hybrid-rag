"""Text-to-SQL backend: SQLite accessed through LangChain's `SQLDatabase`.

Safety is layered so a single failure cannot corrupt data:
  1. `validate()`  - only one SELECT/WITH statement, no DDL/DML keywords
  2. the engine opens the file READ-ONLY (`mode=ro`), so even a query that
     slipped past validation is refused by SQLite itself
  3. a LIMIT is appended when the query has none
  4. a progress handler aborts queries running longer than the timeout
"""

from __future__ import annotations

import logging
import sqlite3
import time
import warnings
from pathlib import Path
from typing import Any

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, event, inspect, text

from app.config import Settings
from app.graph.tables import TableSpec, coerce
from app.text_utils import tokenize

log = logging.getLogger(__name__)

FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "replace", "attach",
    "detach", "pragma", "vacuum", "reindex", "truncate", "grant", "revoke",
    "commit", "rollback", "savepoint",
}


def build_sqlite_database(path: Path, tables: list[TableSpec]) -> None:
    """Write the tables to a fresh SQLite file, with PK and FK constraints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    by_name = {t.name: t for t in tables}

    con = sqlite3.connect(tmp)
    try:
        for t in tables:
            cols = ", ".join(f'"{c}" {t.types[c]}' for c in t.columns)
            fks = "".join(
                f', FOREIGN KEY ("{c}") REFERENCES "{ref}"("{by_name[ref].pk}")'
                for c, ref in t.fks.items() if ref in by_name
            )
            con.execute(f'CREATE TABLE "{t.name}" ({cols}, PRIMARY KEY ("{t.pk}"){fks})')
            marks = ", ".join("?" for _ in t.columns)
            con.executemany(
                f'INSERT OR REPLACE INTO "{t.name}" VALUES ({marks})',
                [[coerce(r.get(c), t.types[c]) for c in t.columns] for r in t.rows],
            )
        con.commit()
    finally:
        con.close()
    # Build into a temp file then swap, so a failed build never leaves a half-written DB.
    tmp.replace(path)
    log.info("Built SQLite database %s (%d tables, %d rows)",
             path, len(tables), sum(len(t.rows) for t in tables))


def _install_query_timeout(engine, seconds: float) -> None:
    """Abort statements that run longer than `seconds` (runaway joins, recursion)."""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, record):
        record.info["deadline"] = None

        def handler() -> int:
            deadline = record.info.get("deadline")
            return 1 if deadline is not None and time.monotonic() > deadline else 0

        dbapi_conn.set_progress_handler(handler, 10_000)

    @event.listens_for(engine, "before_cursor_execute")
    def _before_execute(conn, cursor, statement, parameters, context, executemany):
        conn.connection.info["deadline"] = time.monotonic() + seconds


class SQLiteGraphStore:
    language = "sql"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.sqlite_path
        if not self.path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.path}. Run ingestion first.")
        self.engine = create_engine(f"sqlite:///file:{self.path.as_posix()}?mode=ro&uri=true")
        _install_query_timeout(self.engine, settings.sqlite_query_timeout)
        # sample_rows_in_table_info: LangChain appends example rows to the schema,
        # which shows the model real value formats.
        self.db = SQLDatabase(self.engine, sample_rows_in_table_info=3)
        # Mutually-referencing tables (unit.kepala_pegawai_id <-> pegawai.unit_id)
        # are legitimate but make SQLAlchemy warn while sorting them for the schema.
        warnings.filterwarnings("ignore", message="Cannot correctly sort tables")
        self._inspector = inspect(self.engine)
        self._schema_cache: str | None = None

    # ----------------------------------------------------------------- schema
    def table_names(self) -> list[str]:
        return sorted(self.db.get_usable_table_names())

    def vocabulary(self) -> set[str]:
        words: set[str] = set()
        for table in self.table_names():
            words.update(tokenize(table))
            for col in self._columns(table):
                words.update(t for t in tokenize(col["name"]) if t != "id")
        return words

    def _columns(self, table: str) -> list[dict[str, Any]]:
        return self._inspector.get_columns(table)

    def _foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        out = []
        for fk in self._inspector.get_foreign_keys(table):
            for col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                out.append((col, fk["referred_table"], ref_col))
        return out

    def _primary_key(self, table: str) -> str | None:
        cols = self._inspector.get_pk_constraint(table).get("constrained_columns") or []
        return cols[0] if cols else None

    def _distinct_values(self, table: str, column: str, cap: int = 8) -> list[str] | None:
        """Distinct values of low-cardinality text columns.

        Without them the model guesses casing ('Selesai' vs 'selesai') and
        silently gets zero rows - the most common text-to-SQL failure.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT {cap + 1}'))
            values = [r[0] for r in result.fetchall() if r[0] is not None]
        return sorted(str(v) for v in values) if len(values) <= cap else None

    def schema_description(self) -> str:
        if self._schema_cache is not None:
            return self._schema_cache
        parts = ["SQLite DATABASE SCHEMA", "", self.db.get_table_info().strip(), ""]

        hints: list[str] = []
        for table in self.table_names():
            for col in self._columns(table):
                if "CHAR" in str(col["type"]).upper() or "TEXT" in str(col["type"]).upper():
                    values = self._distinct_values(table, col["name"])
                    if values:
                        hints.append(f'  "{table}"."{col["name"]}" in ({", ".join(repr(v) for v in values)})')
        if hints:
            parts += ["ALLOWED VALUES FOR CATEGORICAL COLUMNS (use exactly as written):", *hints, ""]

        joins = [
            f'  "{table}"."{col}" = "{ref}"."{ref_col}"'
            for table in self.table_names()
            for col, ref, ref_col in self._foreign_keys(table)
        ]
        if joins:
            parts += ["JOIN PATHS (foreign keys):", *joins]
        self._schema_cache = "\n".join(parts).strip()
        return self._schema_cache

    # ------------------------------------------------------------- guarding
    def validate(self, query: str) -> str | None:
        q = (query or "").strip().rstrip(";").strip()
        if not q:
            return "empty query"
        low = q.lower()
        if not low.startswith(("select", "with")):
            return "only SELECT or WITH statements are allowed"
        if ";" in q:
            return "multiple statements are not allowed"
        tokens = set(tokenize(low))
        bad = sorted(tokens & FORBIDDEN)
        if bad:
            return "contains forbidden keywords: " + ", ".join(bad)
        if not tokens & set(self.table_names()):
            return f"does not reference any known table ({', '.join(self.table_names())})"
        return None

    # ------------------------------------------------------------- running
    def run(self, query: str, limit: int = 50) -> tuple[list[str], list[dict[str, Any]]]:
        q = query.strip().rstrip(";").strip()
        if "limit" not in tokenize(q.lower()):
            q = f"{q} LIMIT {limit}"
        # Fetch inside the connection scope; SQLDatabase.run(fetch="cursor") hands
        # back a cursor whose connection has already returned to the pool.
        with self.engine.connect() as conn:
            result = conn.execute(text(q))
            columns = list(result.keys())
            rows = [dict(r) for r in result.mappings().all()]
        return columns, rows

    # ------------------------------------------------------------ few-shot
    def examples(self) -> list[tuple[str, str]]:
        """Few-shot pairs built from YOUR schema, so the model never copies fictional tables."""
        out: list[tuple[str, str]] = []

        def numeric(table: str) -> list[str]:
            pk = self._primary_key(table)
            return [c["name"] for c in self._columns(table)
                    if any(t in str(c["type"]).upper() for t in ("INT", "REAL", "NUM", "FLOAT"))
                    and c["name"] != pk and not c["name"].endswith("_id")]

        def label(table: str) -> str:
            pk = self._primary_key(table)
            texts = [c["name"] for c in self._columns(table)
                     if "TEXT" in str(c["type"]).upper() and c["name"] != pk]
            return texts[0] if texts else (pk or "*")

        for table in self.table_names():
            nums, fks = numeric(table), self._foreign_keys(table)
            if nums and fks:
                col, ref, ref_col = fks[0]
                out.append((
                    f"What is the total {nums[0]} per {ref}?",
                    f'SELECT r."{label(ref)}" AS {ref}, SUM(t."{nums[0]}") AS total_{nums[0]} '
                    f'FROM "{table}" AS t JOIN "{ref}" AS r ON t."{col}" = r."{ref_col}" '
                    f'GROUP BY r."{label(ref)}" ORDER BY total_{nums[0]} DESC',
                ))
                break
        for table in self.table_names():
            nums = numeric(table)
            if nums:
                out.append((
                    f"List the 5 {table} rows with the highest {nums[0]}.",
                    f'SELECT "{label(table)}", "{nums[0]}" FROM "{table}" ORDER BY "{nums[0]}" DESC LIMIT 5',
                ))
                break
        return out

    def close(self) -> None:
        self.engine.dispose()
