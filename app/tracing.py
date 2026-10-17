"""Optional LangSmith tracing, so the pipeline can be observed end to end.

Without this, a wrong answer is a black box: you cannot see which route the
router chose, what the retriever actually returned, what SQL was generated, or
how many times the self-correction loop fired. LangSmith records every nested
LangChain call as a trace tree, which is exactly the evidence the thesis needs
when discussing failure cases.

LangChain reads its tracing configuration from environment variables at call
time, so enabling it means setting those variables before the first chain runs.
That is all this module does - no wrapper, no decorators, nothing in the
pipeline's hot path.

Off by default: tracing sends prompts and retrieved document text to an
external service, which must be a deliberate choice, not a silent one. Set
LANGSMITH_TRACING=true in .env to turn it on.
"""

from __future__ import annotations

import logging
import os

from app.config import Settings

log = logging.getLogger(__name__)

# LangChain also accepts the older LANGCHAIN_* spellings; set both so the
# feature works regardless of the installed version's preference.
_VARS = {
    "tracing": ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
    "api_key": ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    "project": ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
    "endpoint": ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
}


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing when configured. Returns whether it is on.

    Explicitly disables tracing when the setting is off, so a stale
    LANGSMITH_TRACING left in the shell cannot silently start uploading
    prompts.
    """
    if not settings.langsmith_tracing:
        for name in _VARS["tracing"]:
            os.environ[name] = "false"
        return False

    if not settings.langsmith_api_key:
        log.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty; "
                    "tracing stays off. Get a key at https://smith.langchain.com "
                    "(Settings -> API Keys).")
        for name in _VARS["tracing"]:
            os.environ[name] = "false"
        return False

    for name in _VARS["tracing"]:
        os.environ[name] = "true"
    for name in _VARS["api_key"]:
        os.environ[name] = settings.langsmith_api_key
    for name in _VARS["project"]:
        os.environ[name] = settings.langsmith_project
    if settings.langsmith_endpoint:
        for name in _VARS["endpoint"]:
            os.environ[name] = settings.langsmith_endpoint

    log.info("LangSmith tracing enabled (project=%r). Traces: https://smith.langchain.com",
             settings.langsmith_project)
    return True


def tracing_status(settings: Settings) -> dict[str, object]:
    """Tracing state for /health. Never includes the API key."""
    enabled = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
    status: dict[str, object] = {"enabled": enabled, "project": settings.langsmith_project}
    if settings.langsmith_tracing and not settings.langsmith_api_key:
        status["error"] = "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty"
    return status
