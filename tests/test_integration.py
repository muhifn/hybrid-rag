"""Tests that need real infrastructure. Skipped automatically when it is absent.

    Neo4j:   docker compose up -d neo4j   then   pytest tests/test_integration.py
    bge-m3:  RUN_SLOW=1 pytest tests/test_integration.py   (downloads ~2.3 GB once)
"""

from __future__ import annotations

import os

import pytest

from app.config import PROJECT_ROOT, Settings
from app.graph.tables import load_tables


def _neo4j_settings() -> Settings:
    return Settings(_env_file=None, graph_backend="neo4j",
                    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                    neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"))


def _neo4j_available() -> bool:
    try:
        from neo4j import GraphDatabase

        s = _neo4j_settings()
        with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password),
                                  connection_timeout=2) as driver:
            driver.verify_connectivity()
        return True
    except Exception:  # noqa: BLE001
        return False


neo4j_required = pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not reachable")


@neo4j_required
def test_neo4j_backend_end_to_end():
    from app.graph.neo4j_store import Neo4jGraphStore, ingest_into_neo4j

    settings = _neo4j_settings()
    ingest_into_neo4j(settings, load_tables(PROJECT_ROOT / "data" / "tables"))
    store = Neo4jGraphStore(settings)
    try:
        assert {"Pegawai", "Proyek", "Unit"} <= set(store.table_names())
        _, rows = store.run("MATCH (p:Proyek) RETURN sum(p.anggaran) AS total")
        assert rows[0]["total"] == 2_199_000_000
        _, rows = store.run("MATCH (p:Proyek)-[:UNIT]->(u:Unit) RETURN u.nama_unit AS unit, "
                            "sum(p.anggaran) AS total ORDER BY total DESC LIMIT 1")
        assert rows[0] == {"unit": "Operasional", "total": 785_000_000}
        assert store.validate("MATCH (n) DETACH DELETE n") is not None
        with pytest.raises(Exception):  # READ session: the server refuses writes
            store.run("MATCH (n:Unit) SET n.x = 1 RETURN n")
        for _, cypher in store.examples():
            store.run(cypher)
    finally:
        store.close()


@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to load BAAI/bge-m3")
def test_bge_m3_retrieves_indonesian_paraphrases(tmp_path):
    """Real embeddings: questions share almost no words with the matching passage."""
    from app.llm import build_embeddings
    from app.service import RAGService
    from tests.conftest import make_settings
    from tests.fakes import FakeChatModel

    settings = make_settings(tmp_path, embedding_model="BAAI/bge-m3", top_k=1)
    svc = RAGService(settings, llm=FakeChatModel(), embeddings=build_embeddings(settings))
    try:
        svc.ingest(recreate=True)
        assert svc.dimension == 1024
        cases = {
            "berapa hari jatah libur karyawan dalam setahun?": "kebijakan-cuti.md",
            "kapan aplikasi tidak bisa dibuka karena perawatan?": "faq-sistem.md",
            "pembelian bernilai ratusan juta harus lewat lelang?": "panduan-pengadaan.md",
        }
        for question, expected in cases.items():
            top = svc.retrievers.retrieve(question, mode="dense")[0]
            assert top.metadata["source"] == expected, question
    finally:
        svc.close()
