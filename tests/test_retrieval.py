from __future__ import annotations

import pytest

from app.router import QueryRouter
from app.vectorstore import build_retrievers
from tests.fakes import GarbageChatModel


def test_every_chunk_is_indexed(service):
    assert service.index.count() == len(service.index.all_documents()) > 0


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_return_at_most_top_k(service, mode):
    docs = service.retrievers.retrieve("cuti tahunan pegawai", mode=mode)
    assert 0 < len(docs) <= service.settings.top_k


def test_bm25_finds_rare_acronym(service):
    # "KAK" (Kerangka Acuan Kerja) only appears in the procurement guide.
    docs = service.retrievers.retrieve("KAK RAB", mode="bm25")
    assert docs[0].metadata["source"] == "panduan-pengadaan.md"


def test_bm25_is_case_insensitive(service):
    upper = [d.metadata["chunk_id"] for d in service.retrievers.retrieve("CUTI TAHUNAN", mode="bm25")]
    lower = [d.metadata["chunk_id"] for d in service.retrievers.retrieve("cuti tahunan", mode="bm25")]
    assert upper == lower


def test_hybrid_results_are_deduplicated(service):
    ids = [d.metadata["chunk_id"] for d in service.retrievers.retrieve("cuti", mode="hybrid")]
    assert len(ids) == len(set(ids))


def test_reingest_without_recreate_does_not_duplicate(fresh_service):
    before = fresh_service.index.count()
    fresh_service.ingest(recreate=False)
    assert fresh_service.index.count() == before


def test_empty_store_falls_back_to_dense(fresh_service):
    fresh_service.index.ensure_collection(recreate=True)
    retrievers = build_retrievers(fresh_service.index, fresh_service.settings)
    assert retrievers.bm25 is None and retrievers.hybrid is retrievers.dense


def test_reingest_drops_chunks_of_removed_documents(tmp_path):
    """Regression: embedded Qdrant on Windows used to resurrect deleted points."""
    import shutil

    from app.config import PROJECT_ROOT
    from app.service import RAGService
    from tests.conftest import make_settings
    from tests.fakes import FakeChatModel, HashingEmbeddings

    docs = tmp_path / "docs"
    shutil.copytree(PROJECT_ROOT / "data" / "documents", docs)
    svc = RAGService(make_settings(tmp_path, documents_dir=docs), llm=FakeChatModel(),
                     embeddings=HashingEmbeddings())
    try:
        svc.ingest(recreate=True)
        assert "faq-sistem.md" in {d.metadata["source"] for d in svc.index.all_documents()}
        (docs / "faq-sistem.md").unlink()
        svc.ingest(recreate=True)
        sources = {d.metadata["source"] for d in svc.index.all_documents()}
        assert sources == {"kebijakan-cuti.md", "panduan-pengadaan.md"}
        assert all(d.metadata["source"] != "faq-sistem.md" for d in svc.retrievers.retrieve("password", "bm25"))
    finally:
        svc.close()


def test_recreate_handles_embedding_dimension_change(fresh_service):
    index = fresh_service.index
    index.dimension = 64  # as if the embedding model had been swapped
    index.ensure_collection(recreate=True)
    assert index.count() == 0
    assert index.client.get_collection(index.collection).config.params.vectors.size == 64


# ------------------------------------------------------------------ router
def test_router_uses_structured_output(service):
    decision = service.router.classify("Berapa total anggaran seluruh proyek?")
    assert decision.route == "GRAPH" and decision.method == "llm"


def test_router_falls_back_to_heuristic_on_garbage():
    router = QueryRouter(GarbageChatModel(), ["proyek"], {"anggaran", "proyek"}, ["Panduan"])
    decision = router.classify("Berapa total anggaran proyek?")
    assert decision.method == "fallback" and decision.route == "GRAPH"


@pytest.mark.parametrize("question,expected", [
    ("Bagaimana prosedur pengajuan cuti?", "VECTOR"),
    ("Berapa total anggaran proyek?", "GRAPH"),
    ("Proyek mana yang wajib tender, sebutkan anggarannya", "HYBRID"),
    ("halo", "VECTOR"),
])
def test_heuristic(question, expected):
    router = QueryRouter(GarbageChatModel(), ["proyek"], {"anggaran", "proyek"}, [])
    assert router.heuristic(question).route == expected


def test_router_accuracy_on_gold_set(service, gold):
    correct = sum(service.router.classify(g["question"]).route == g["route"] for g in gold)
    assert correct == len(gold)
