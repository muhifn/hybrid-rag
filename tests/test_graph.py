from __future__ import annotations

import pytest

from app.graph.query_engine import GraphQueryEngine
from app.graph.sqlite_store import SQLiteGraphStore
from app.text_utils import strip_code_fences
from tests.conftest import make_settings
from tests.fakes import FakeChatModel


def test_aggregates_are_exact(service):
    store = service.graph_store
    _, rows = store.run("SELECT SUM(anggaran) AS total FROM proyek")
    assert rows[0]["total"] == 2_199_000_000
    _, rows = store.run(
        "SELECT u.nama_unit, SUM(p.anggaran) AS total FROM proyek p "
        "JOIN unit u ON p.unit_id = u.unit_id GROUP BY u.nama_unit ORDER BY total DESC")
    assert rows[0] == {"nama_unit": "Operasional", "total": 785_000_000}


@pytest.mark.parametrize("query", [
    "DROP TABLE pegawai",
    "DELETE FROM pegawai",
    "UPDATE pegawai SET gaji_pokok = 0",
    "INSERT INTO unit VALUES ('x','y','z','w')",
    "SELECT 1; DROP TABLE pegawai",
    "SELECT 1",  # references no known table
    "PRAGMA table_info(pegawai)",
    "",
])
def test_validator_rejects_unsafe_queries(service, query):
    assert service.graph_store.validate(query) is not None


def test_validator_accepts_plain_select(service):
    assert service.graph_store.validate("SELECT nama FROM pegawai WHERE gaji_pokok > 9000000") is None


def test_engine_is_read_only_even_if_validation_is_bypassed(service):
    from sqlalchemy import text

    # Go straight to the engine: no validator, no auto-LIMIT.
    with pytest.raises(Exception, match="readonly|read-only|attempt to write"):
        with service.graph_store.engine.begin() as conn:
            conn.execute(text("DELETE FROM pegawai"))
    _, rows = service.graph_store.run("SELECT COUNT(*) AS n FROM pegawai")
    assert rows[0]["n"] == 12


def test_runaway_query_is_interrupted(fresh_service, tmp_path):
    settings = make_settings(tmp_path, sqlite_query_timeout=0.5,
                             sqlite_path=fresh_service.settings.sqlite_path)
    store = SQLiteGraphStore(settings)
    try:
        with pytest.raises(Exception, match="interrupt"):
            store.run("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
                      "SELECT COUNT(*) FROM c, proyek")
    finally:
        store.close()


def test_schema_lists_categorical_values_and_join_paths(service):
    schema = service.graph_store.schema_description()
    assert "'berjalan'" in schema and "'selesai'" in schema
    assert '"proyek"."unit_id" = "unit"."unit_id"' in schema


def test_synthesised_examples_execute(service):
    examples = service.graph_store.examples()
    assert examples
    for _, sql in examples:
        assert service.graph_store.validate(sql) is None
        service.graph_store.run(sql)


def test_every_reference_query_in_gold_set_executes(service, gold):
    for item in gold:
        if item.get("ref_sql"):
            service.graph_store.run(item["ref_sql"], limit=200)


def test_self_correction_recovers_on_second_attempt(service):
    engine = GraphQueryEngine(service.graph_store, service.llm, max_attempts=3)
    result = engine.ask("self_correct_me: berapa proyek yang statusnya berjalan")
    assert result.ok and result.attempts == 2
    assert result.rows == [{"jumlah": 5}]


def test_engine_gives_up_after_max_attempts(service):
    stubborn = FakeChatModel(responder=lambda m: "SELECT * FROM tabel_tidak_ada")
    engine = GraphQueryEngine(service.graph_store, stubborn, max_attempts=2)
    result = engine.ask("apa saja")
    assert not result.ok and result.attempts == 2
    assert "failed after 2 attempts" in result.error


@pytest.mark.parametrize("raw,expected", [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("SQL: SELECT 1;", "SELECT 1"),
    ("```cypher\nMATCH (n) RETURN n\n```", "MATCH (n) RETURN n"),
])
def test_code_fences_are_stripped(raw, expected):
    assert strip_code_fences(raw) == expected
