"""Application settings, loaded from environment variables and `.env`.

Every tunable of the system lives here so experiments are reproducible: the
values used for a run can be logged verbatim (see `Settings.public_dict`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Chat LLM: LM Studio through LangChain's ChatOpenAI ---------------
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"  # LM Studio ignores it; the SDK requires one
    llm_model: str = ""  # blank -> first chat model loaded in LM Studio
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: float = 180.0
    llm_max_retries: int = 2
    router_structured_output: bool = True  # use JSON-schema constrained decoding

    # ---- Embeddings --------------------------------------------------------
    # backend="huggingface": run the model in-process (needs the RAM/VRAM).
    # backend="openai":      LM Studio serves it at LLM_BASE_URL + /embeddings.
    # On a ~8 GB machine, serving embeddings from LM Studio frees the RAM that
    # an in-process bge-m3 (~2.2 GB) would otherwise need next to the chat LLM.
    embedding_backend: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "BAAI/bge-m3"
    # "cpu" by default on purpose: on a single consumer GPU, LM Studio needs the
    # VRAM for the chat model. Use "cuda" only with a CUDA build of torch and
    # enough VRAM for both models. Ignored when embedding_backend="openai".
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    hf_cache_dir: str | None = None

    # ---- Vector store: Qdrant ----------------------------------------------
    qdrant_mode: Literal["embedded", "server"] = "embedded"
    qdrant_path: Path = PROJECT_ROOT / "storage" / "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "thesis_docs"

    # ---- Structured data: switchable graph backend -------------------------
    graph_backend: Literal["sqlite", "neo4j"] = "sqlite"
    sqlite_path: Path = PROJECT_ROOT / "storage" / "graph.db"
    sqlite_query_timeout: float = 10.0
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # ---- Source data -------------------------------------------------------
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    tables_dir: Path = PROJECT_ROOT / "data" / "tables"
    # Chunks from the OCR pipeline (`python -m scripts.ocr_ingest`). When this
    # file exists it is preferred over documents_dir: those chunks carry page
    # numbers and section paths, and keep tables whole - boundaries the
    # character splitter here cannot reproduce. Delete the file (or set
    # USE_OCR_CHUNKS=false) to fall back to plain documents_dir loading.
    ocr_chunks_file: Path = PROJECT_ROOT / "data" / "ocr_out" / "jsonl" / "semua-chunk.jsonl"
    use_ocr_chunks: bool = True
    # For CSV columns that do not follow the `<table>_id` convention. JSON in .env:
    #   FK_OVERRIDES={"proyek.penanggung_jawab_id": "pegawai"}
    fk_overrides: dict[str, str] = Field(default_factory=dict)

    # ---- Retrieval ---------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    retrieval_candidates: int = 20  # per-retriever pool before RRF fusion
    bm25_weight: float = 0.5  # dense weight is 1 - bm25_weight
    rrf_c: int = 60  # Cormack et al. (2009)

    # ---- Text-to-query -----------------------------------------------------
    graph_max_attempts: int = 3
    graph_row_limit: int = 50

    # ---- Conversation memory -----------------------------------------------
    max_history_turns: int = 6
    max_sessions: int = 500

    # ---- Observability: LangSmith ------------------------------------------
    # Off by default: tracing uploads prompts and retrieved document text to an
    # external service, so it has to be an explicit choice. When on, every
    # router decision, retrieval, generated SQL, and self-correction retry
    # shows up as a trace tree at https://smith.langchain.com
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "hybrid-rag-api"
    langsmith_endpoint: str | None = None  # only for self-hosted LangSmith

    # ---- API ---------------------------------------------------------------
    admin_api_key: str | None = None  # protects /admin/* when set
    cors_origins: str = ""  # comma-separated, e.g. "http://localhost:8501"

    @field_validator(
        "qdrant_path", "sqlite_path", "documents_dir", "tables_dir", "ocr_chunks_file", mode="after"
    )
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        # Relative paths in .env are relative to the project, not to the
        # directory uvicorn happens to be started from.
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info) -> int:
        size = info.data.get("chunk_size", 800)
        if value >= size:
            raise ValueError(f"chunk_overlap ({value}) must be smaller than chunk_size ({size})")
        return value

    @field_validator("bm25_weight", mode="after")
    @classmethod
    def _weight_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("bm25_weight must be between 0 and 1")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_lm_studio(self) -> bool:
        return "api.openai.com" not in self.llm_base_url

    def public_dict(self) -> dict:
        """Settings safe to log or return from the API (no secrets)."""
        hidden = {"llm_api_key", "qdrant_api_key", "neo4j_password", "admin_api_key",
                  "langsmith_api_key"}
        return {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in self.model_dump().items()
            if k not in hidden
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
