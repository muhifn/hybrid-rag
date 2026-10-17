from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.service import RAGService
from tests.fakes import FakeChatModel, HashingEmbeddings, scripted_reply

GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.json"


def make_settings(tmp: Path, **overrides) -> Settings:
    """Settings isolated from the user's .env and pointed at throwaway storage."""
    values = dict(
        llm_model="fake",
        documents_dir=PROJECT_ROOT / "data" / "documents",
        tables_dir=PROJECT_ROOT / "data" / "tables",
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
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


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
