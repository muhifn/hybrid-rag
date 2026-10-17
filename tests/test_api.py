from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.service import RAGService
from tests.conftest import make_settings
from tests.fakes import FakeChatModel, HashingEmbeddings, scripted_reply


@pytest.fixture()
def client(tmp_path, gold):
    def factory():
        svc = RAGService(make_settings(tmp_path, admin_api_key="secret"),
                         llm=FakeChatModel(responder=lambda m: scripted_reply(m, gold)),
                         embeddings=HashingEmbeddings())
        svc.ingest(recreate=True)
        return svc

    with TestClient(create_app(factory)) as c:
        yield c


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(client):
    body = client.get("/health").json()
    assert body["ready"] is True
    assert body["vector_store"]["chunks"] > 0
    assert body["graph"]["available"] and "proyek" in body["graph"]["tables"]


def test_chat_returns_answer_sources_and_session(client):
    r = client.post("/chat", json={"question": "Proyek mana yang wajib menggunakan tender terbuka?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "HYBRID" and body["session_id"]
    assert body["sources"][0]["source"].endswith(".md")
    assert body["graph"]["ok"] and body["graph"]["language"] == "sql"


def test_session_memory_roundtrip(client):
    sid = client.post("/chat", json={"question": "Berapa hari cuti tahunan?"}).json()["session_id"]
    second = client.post("/chat", json={"question": "Kalau tidak dipakai bagaimana?", "session_id": sid}).json()
    assert second["standalone_question"] != second["question"]
    history = client.get(f"/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert client.delete(f"/sessions/{sid}").json()["deleted"] is True
    assert client.get(f"/sessions/{sid}").json()["messages"] == []


def test_stream_emits_sse_events(client):
    r = client.post("/chat/stream", json={"question": "Bagaimana prosedur cuti?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0][0] == "meta" and events[-1][0] == "done"
    tokens = "".join(d["text"] for e, d in events if e == "token")
    assert events[-1][1]["answer"] == tokens


def test_validation_errors(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "x", "route": "NOPE"}).status_code == 422


def test_admin_requires_key(client):
    assert client.post("/admin/ingest").status_code == 401
    r = client.post("/admin/ingest", headers={"X-Admin-Key": "secret"}, json={"recreate": True})
    assert r.status_code == 200 and r.json()["documents"] == 3
    schema = client.get("/admin/schema", headers={"X-Admin-Key": "secret"}).json()
    assert schema["language"] == "sql" and "proyek" in schema["schema"]


def test_chat_before_ingestion_returns_503(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    def factory():
        return RAGService(make_settings(tmp_path, documents_dir=empty, tables_dir=empty),
                          llm=FakeChatModel(), embeddings=HashingEmbeddings())

    with TestClient(create_app(factory)) as c:
        assert c.get("/health").json()["ready"] is False
        assert c.post("/chat", json={"question": "halo"}).status_code == 503


def _down_client(tmp_path, exc):
    class DownChatModel(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise exc

    def factory():
        svc = RAGService(make_settings(tmp_path), llm=DownChatModel(), embeddings=HashingEmbeddings())
        svc.ingest(recreate=True)
        return svc

    return TestClient(create_app(factory))


def test_llm_down_maps_to_503(tmp_path):
    import httpx
    import openai

    exc = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:1234/v1/chat/completions"))
    with _down_client(tmp_path, exc) as c:
        r = c.post("/chat", json={"question": "Berapa hari cuti tahunan?"})
        assert r.status_code == 503
        assert "Cannot reach the LLM server" in r.json()["detail"]


def test_llm_auth_failure_maps_to_503_with_hint(tmp_path):
    import httpx
    import openai

    request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
    response = httpx.Response(401, request=request)
    exc = openai.AuthenticationError("401", response=response, body=None)
    with _down_client(tmp_path, exc) as c:
        r = c.post("/chat", json={"question": "Berapa hari cuti tahunan?"})
        assert r.status_code == 503
        assert "LLM_API_KEY" in r.json()["detail"]


def test_stream_llm_down_reports_error_event(tmp_path):
    import httpx
    import openai

    exc = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:1234/v1/chat/completions"))
    with _down_client(tmp_path, exc) as c:
        r = c.post("/chat/stream", json={"question": "Berapa hari cuti tahunan?"})
        assert r.status_code == 200  # SSE: failure arrives in-band
        events = parse_sse(r.text)
        assert events[-1][0] == "error"
        assert "Cannot reach the LLM server" in events[-1][1]["detail"]
