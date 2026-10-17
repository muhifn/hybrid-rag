"""Model factories: the chat LLM (LM Studio via ChatOpenAI) and the embedding model."""

from __future__ import annotations

import logging

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings
from app.text_utils import strip_reasoning

log = logging.getLogger(__name__)

_EMBEDDING_NAME_HINTS = ("embed", "bge", "e5-", "gte-", "minilm", "nomic")


def list_server_models(settings: Settings, timeout: float = 5.0) -> list[str]:
    """Return the model ids currently served by the OpenAI-compatible endpoint."""
    url = settings.llm_base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", [])]


def resolve_model_name(settings: Settings) -> str:
    """Use LLM_MODEL if set; otherwise pick the first loaded chat model."""
    if settings.llm_model:
        return settings.llm_model
    try:
        models = list_server_models(settings)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise RuntimeError(
                "LM Studio rejected the API key (HTTP 401). Your server has 'Require API token' "
                "enabled: create a token in LM Studio (Developer -> Server Settings) and put it in "
                "LLM_API_KEY in .env."
            ) from exc
        raise
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"LLM_MODEL is empty and the LLM server at {settings.llm_base_url} is "
            f"unreachable ({exc}). Start the LM Studio server (Developer tab -> "
            "Start Server) or set LLM_MODEL explicitly in .env."
        ) from exc
    chat_models = [m for m in models if not any(h in m.lower() for h in _EMBEDDING_NAME_HINTS)]
    if not chat_models:
        raise RuntimeError(
            f"No chat model is loaded in LM Studio (server reports: {models}). "
            "Load an instruct model, e.g. qwen2.5-7b-instruct."
        )
    log.info("LLM_MODEL not set; using first loaded chat model: %s", chat_models[0])
    return chat_models[0]


def build_chat_model(
    settings: Settings,
    model_name: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """ChatOpenAI pointed at LM Studio.

    langchain-openai 1.x serialises `max_tokens` as `max_completion_tokens`,
    which LM Studio does not read - generation would silently run unbounded.
    For LM Studio we therefore send `max_tokens` through `extra_body`; for the
    real OpenAI API we use the normal field.
    """
    limit = settings.llm_max_tokens if max_tokens is None else max_tokens
    kwargs: dict = {
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "model": model_name,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }
    if settings.is_lm_studio:
        kwargs["extra_body"] = {"max_tokens": limit}
    else:
        kwargs["max_tokens"] = limit
    return ChatOpenAI(**kwargs)


def text_parser() -> Runnable:
    """StrOutputParser that also drops inline <think> blocks from reasoning models."""
    return StrOutputParser() | RunnableLambda(strip_reasoning)


def build_embeddings(settings: Settings) -> Embeddings:
    """Build the embedding function for the configured backend.

    BAAI/bge-m3: multilingual (incl. Bahasa Indonesia), 1024-dim, 8192-token input.

    backend="huggingface" runs the model in-process via sentence-transformers
    (`normalize_embeddings=True` makes vectors unit-length so cosine similarity
    in Qdrant behaves as the model card intends). backend="openai" instead calls
    the embedding model served by LM Studio at LLM_BASE_URL + /embeddings -
    which frees the RAM/VRAM an in-process bge-m3 would otherwise occupy next to
    the chat model. The returned object is a LangChain `Embeddings` either way,
    so Qdrant and the rest of the pipeline never know the difference.
    """
    if settings.embedding_backend == "openai":
        log.info("Using embedding model %s served at %s/embeddings",
                 settings.embedding_model, settings.llm_base_url.rstrip("/"))
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            # bge-m3 has no tiktoken encoding; without this OpenAIEmbeddings
            # would try to length-check/truncate with tiktoken and fail.
            check_embedding_ctx_length=False,
        )
    log.info("Loading embedding model %s on %s (first run downloads ~2.3 GB)",
             settings.embedding_model, settings.embedding_device)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
        cache_folder=settings.hf_cache_dir,
    )


def embedding_dimension(embeddings: Embeddings) -> int:
    """Measure the vector size empirically instead of hard-coding it."""
    return len(embeddings.embed_query("dimension probe"))
