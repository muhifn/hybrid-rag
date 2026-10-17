"""Reasoning models (Qwen3.x etc.) and LM Studio authentication."""

from __future__ import annotations

import re

import httpx
import pytest

from app.config import Settings
from app.llm import resolve_model_name
from app.pipeline import EMPTY_ANSWER
from app.text_utils import extract_json, filter_reasoning_stream, strip_code_fences, strip_reasoning
from tests.fakes import FakeChatModel, scripted_reply


@pytest.mark.parametrize("raw,expected", [
    ("<think>hmm</think>\n\nJawaban.", "Jawaban."),
    ("<THINK>a\nb</THINK>Jawaban", "Jawaban"),
    ("Tanpa reasoning.", "Tanpa reasoning."),
    ("<think>cut off by the token limit", ""),
])
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_queries_and_json_survive_think_blocks():
    assert strip_code_fences("<think>which table?</think>\n```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_json('<think>{"route": "GRAPH"} maybe?</think>{"route": "VECTOR"}') == {"route": "VECTOR"}


@pytest.mark.parametrize("tokens,expected", [
    (["<thi", "nk>", "reason", "ing</th", "ink>", "\n\n", "Jawa", "ban."], "Jawaban."),
    (["  <think>x</think>Halo"], "Halo"),
    (["Ha", "lo ", "dunia"], "Halo dunia"),
    (["<", "b>tebal</b>"], "<b>tebal</b>"),  # starts like a tag, but is not <think>
    (["<thi"], "<thi"),  # stream ends while undecided: emit what we have
    (["<think>never closed"], ""),
])
def test_filter_reasoning_stream(tokens, expected):
    assert "".join(filter_reasoning_stream(tokens)) == expected


def _thinking_model(gold, answer_tail="Jawaban akhir [1]."):
    def reply(messages):
        text = scripted_reply(messages, gold)
        if messages[0].content.startswith("You answer questions"):
            text = answer_tail
        return f"<think>Let me reason about this carefully.</think>\n\n{text}"

    return FakeChatModel(responder=reply)


@pytest.fixture()
def thinking_service(fresh_service, gold):
    from app.pipeline import HybridRAGPipeline

    fresh_service.llm = _thinking_model(gold)
    fresh_service._load_components()
    assert isinstance(fresh_service.pipeline, HybridRAGPipeline)
    return fresh_service


def test_whole_pipeline_with_a_thinking_model(thinking_service):
    result = thinking_service.ask("Berapa total anggaran proyek untuk setiap unit?")
    assert result.route == "GRAPH" and result.route_method == "llm"
    assert result.graph.ok and result.graph.attempts == 1  # SQL parsed despite <think>
    assert result.answer == "Jawaban akhir [1]."


def test_streaming_hides_reasoning(thinking_service):
    events = list(thinking_service.stream("Bagaimana prosedur cuti?"))
    streamed = "".join(p for k, p in events if k == "token")
    assert "<think>" not in streamed and streamed == "Jawaban akhir [1]."


def test_reasoning_that_eats_the_token_budget_is_explained(fresh_service, gold):
    def reply(messages):
        if messages[0].content.startswith("You answer questions"):
            return "<think>still thinking when max_tokens ran out"
        return scripted_reply(messages, gold)

    fresh_service.llm = FakeChatModel(responder=reply)
    fresh_service._load_components()
    assert fresh_service.ask("Bagaimana prosedur cuti?").answer == EMPTY_ANSWER
    streamed = "".join(p for k, p in fresh_service.stream("Bagaimana prosedur cuti?") if k == "token")
    assert streamed == EMPTY_ANSWER


def test_missing_lm_studio_token_gives_a_clear_error(monkeypatch):
    request = httpx.Request("GET", "http://localhost:1234/v1/models")
    response = httpx.Response(401, request=request)

    def unauthorized(*args, **kwargs):
        raise httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)

    monkeypatch.setattr("app.llm.list_server_models", unauthorized)
    with pytest.raises(RuntimeError, match=re.escape("LLM_API_KEY")):
        resolve_model_name(Settings(_env_file=None, llm_model=""))
