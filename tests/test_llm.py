from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.config import Settings
from app.llm import build_chat_model, build_embeddings, resolve_model_name


def test_lm_studio_gets_max_tokens_in_body():
    # langchain-openai 1.x sends `max_completion_tokens`, which LM Studio ignores.
    settings = Settings(_env_file=None, llm_base_url="http://localhost:1234/v1", llm_max_tokens=321)
    llm = build_chat_model(settings, "qwen2.5-7b-instruct")
    assert llm.extra_body == {"max_tokens": 321}
    payload = llm._get_request_payload([HumanMessage("hi")])
    assert "max_completion_tokens" not in payload


def test_openai_uses_the_standard_field():
    settings = Settings(_env_file=None, llm_base_url="https://api.openai.com/v1", llm_max_tokens=321)
    llm = build_chat_model(settings, "gpt-4o-mini")
    assert llm.extra_body is None
    assert llm._get_request_payload([HumanMessage("hi")])["max_completion_tokens"] == 321


def test_explicit_model_needs_no_server():
    settings = Settings(_env_file=None, llm_model="my-model", llm_base_url="http://127.0.0.1:9/v1")
    assert resolve_model_name(settings) == "my-model"


def test_blank_model_with_server_down_fails_clearly():
    settings = Settings(_env_file=None, llm_model="", llm_base_url="http://127.0.0.1:9/v1")
    with pytest.raises(RuntimeError, match="Start the LM Studio server"):
        resolve_model_name(settings)


def test_relative_paths_resolve_against_project_root():
    from app.config import PROJECT_ROOT

    settings = Settings(_env_file=None, qdrant_path="storage/q")
    assert settings.qdrant_path == (PROJECT_ROOT / "storage" / "q").resolve()


def test_invalid_chunk_overlap_is_rejected():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_openai_embedding_backend_uses_lm_studio_endpoint():
    from langchain_openai import OpenAIEmbeddings

    settings = Settings(
        _env_file=None,
        embedding_backend="openai",
        embedding_model="text-embedding-bge-m3",
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="lm-studio",
    )
    emb = build_embeddings(settings)
    assert isinstance(emb, OpenAIEmbeddings)
    # No tiktoken length check: bge-m3 has no tiktoken encoding.
    assert emb.check_embedding_ctx_length is False
    # The model id is passed through to the request.
    assert emb.model == "text-embedding-bge-m3"


def test_default_embedding_backend_is_huggingface():
    settings = Settings(_env_file=None)
    assert settings.embedding_backend == "huggingface"
