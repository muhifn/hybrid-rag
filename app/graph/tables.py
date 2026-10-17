"""Read CSV files into table specs and infer types and foreign keys.

Conventions (so your own CSVs need no code changes):
  * one CSV file = one table (SQLite) / one node label (Neo4j)
  * the FIRST column is the primary key
  * a column ending in `_id` is a foreign key when it equals, or ends with
    `_<pk>`, the primary key of another table:
        unit_id                      -> table `unit`     (exact match)
        penanggung_jawab_pegawai_id  -> table `pegawai`  (suffix match)
        penanggung_jawab_id          -> NOT detected; declare it in fk_overrides
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TableSpec:
    name: str
    columns: list[str]
    types: dict[str, str]  # column -> INTEGER | REAL | TEXT
    pk: str
    rows: list[dict[str, Any]]
    fks: dict[str, str] = field(default_factory=dict)  # column -> referenced table


def _is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def infer_type(values: list[str | None]) -> str:
    present = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not present:
        return "TEXT"
    if all(_is_int(v) for v in present):
        return "INTEGER"
    if all(_is_float(v) for v in present):
        return "REAL"
    return "TEXT"


def coerce(value: Any, sql_type: str) -> Any:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        if sql_type == "INTEGER":
            return int(text)
        if sql_type == "REAL":
            return float(text)
    except ValueError:
        return text
    return text


def table_name_for(path: Path) -> str:
    return path.stem.strip().lower().replace("-", "_").replace(" ", "_")


def load_tables(tables_dir: Path, fk_overrides: dict[str, str] | None = None) -> list[TableSpec]:
    """Load every CSV under `tables_dir`.

    `fk_overrides` maps "table.column" -> referenced table for columns that do
    not follow the naming convention, e.g. {"proyek.penanggung_jawab_id": "pegawai"}.
    """
    files = sorted(tables_dir.rglob("*.csv")) if tables_dir.exists() else []
    if not files:
        log.warning("No CSV files in %s — graph backend will be unavailable", tables_dir)
        return []

    specs: list[TableSpec] = []
    for path in files:
        # utf-8-sig strips the BOM Excel likes to write
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            columns = [c.strip() for c in (reader.fieldnames or []) if c and c.strip()]
            rows = [{k.strip(): v for k, v in r.items() if k} for r in reader]
        if not columns or not rows:
            log.warning("Skipping %s: no header or no rows", path.name)
            continue
        types = {c: infer_type([r.get(c) for r in rows]) for c in columns}
        specs.append(TableSpec(name=table_name_for(path), columns=columns, types=types,
                               pk=columns[0], rows=rows))

    pk_by_table = {s.name: s.pk for s in specs}
    overrides = fk_overrides or {}
    for spec in specs:
        for col in spec.columns:
            manual = overrides.get(f"{spec.name}.{col}")
            if manual:
                if manual in pk_by_table:
                    spec.fks[col] = manual
                else:
                    log.warning("fk_overrides points %s.%s at unknown table %r", spec.name, col, manual)
                continue
            if col == spec.pk or not col.endswith("_id"):
                continue
            for other, other_pk in pk_by_table.items():
                if other != spec.name and (col == other_pk or col.endswith("_" + other_pk)):
                    spec.fks[col] = other
                    break

    for s in specs:
        log.info("Table %-12s %4d rows | pk=%s | fks=%s", s.name, len(s.rows), s.pk, s.fks or "-")
    return specs
