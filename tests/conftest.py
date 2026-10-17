from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.service import RAGService
from tests.fakes import FakeChatModel, HashingEmbeddings, scripted_reply

# Tests run against a small fixed corpus, NOT against data/. The real corpus
# is hundreds of pages of academic documents: too slow to ingest per test run,
# and its content changes whenever the user re-ingests, which would make
# assertions about retrieval results rot. These three documents and three CSVs
# are the former demo data, kept here precisely because they are stable.
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_GOLD = FIXTURES / "gold_set.json"


def make_settings(tmp: Path, **overrides) -> Settings:
    """Settings isolated from the user's .env and pointed at throwaway storage."""
    values = dict(
        llm_model="fake",
        documents_dir=FIXTURES / "documents",
        tables_dir=FIXTURES / "tables",
        # The fixture corpus is plain markdown, so never pick up the OCR
        # chunks a developer may have generated in data/ocr_out/.
        use_ocr_chunks=False,
        qdrant_mode="embedded",
        qdrant_path=tmp / "qdrant",
        qdrant_collection="test_docs",
        graph_backend="sqlite",
        sqlite_path=tmp / "graph.db",
        top_k=3,
        retrieval_candidates=10,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="session")
def gold() -> list[dict]:
    """The gold set matching the fixture corpus.

    Deliberately not eval/gold_set.json: that one asks about the real academic
    documents, so its questions and reference SQL do not apply to the three
    fixture documents the tests ingest.
    """
    return json.loads(FIXTURE_GOLD.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def service(tmp_path_factory, gold):
    """One ingested service shared by read-only tests."""
    llm = FakeChatModel(responder=lambda m: scripted_reply(m, gold))
    svc = RAGService(make_settings(tmp_path_factory.mktemp("svc")), llm=llm, embeddings=HashingEmbeddings())
    svc.ingest(recreate=True)
    yield svc
    svc.close()


@pytest.fixture()
def fresh_service(tmp_path, gold):
    """A private service for tests that mutate state."""
    llm = FakeChatModel(responder=lambda m: scripted_reply(m, gold))
    svc = RAGService(make_settings(tmp_path), llm=llm, embeddings=HashingEmbeddings())
    svc.ingest(recreate=True)
    yield svc
    svc.close()
