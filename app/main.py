"""FastAPI application.

Run:   uvicorn app.main:app --host 127.0.0.1 --port 8000
Docs:  http://127.0.0.1:8000/docs

Use a single worker (the default). In embedded mode Qdrant locks its folder to
one process; for multiple workers, switch to QDRANT_MODE=server.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import openai

from app.config import get_settings
from app.schemas import (
    ChatRequest, ChatResponse, HistoryMessage, HistoryResponse, IngestRequest, IngestResponse,
)
from app.service import NotReadyError, RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("app")

# langchain_openai.chat_models.base.OpenAIConnectionError subclasses
# openai.APIConnectionError, so catching the openai class covers both.
CONNECTION_HINT = (
    "Cannot reach the LLM server. Start it in LM Studio (Developer tab -> Start Server) "
    "and check LLM_BASE_URL (the /v1 suffix is required)."
)
AUTH_HINT = (
    "The LLM server rejected the API key (401). If LM Studio has 'Require API token' enabled, "
    "create a token there and put it in LLM_API_KEY in .env."
)


def _llm_http_error(exc: Exception) -> HTTPException | None:
    """Map LLM infrastructure failures to a clean 503 instead of a raw 500."""
    if isinstance(exc, openai.AuthenticationError):
        return HTTPException(status_code=503, detail=AUTH_HINT)
    if isinstance(exc, openai.APIConnectionError):
        return HTTPException(status_code=503, detail=CONNECTION_HINT)
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 401:
        return HTTPException(status_code=503, detail=AUTH_HINT)
    return None


def create_app(service_factory: Callable[[], RAGService] | None = None) -> FastAPI:
    """App factory. Tests pass a factory that builds the service with fake models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory() if service_factory else RAGService(get_settings())
        log.info("Service ready: %s", {k: v for k, v in app.state.service.health().items()
                                         if k in ("ready", "vector_store", "graph")})
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(
        title="Hybrid RAG API",
        description="Vector (Qdrant + BM25) and graph (SQL / Cypher) retrieval with a local LLM.",
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = get_settings().cors_origin_list
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def service(request: Request) -> RAGService:
        return request.app.state.service

    def require_admin(request: Request, x_admin_key: str | None = Header(default=None)) -> None:
        expected = request.app.state.service.settings.admin_api_key
        if expected and x_admin_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key header")

    # ------------------------------------------------------------- routes
    @app.get("/", tags=["meta"])
    def root():
        return {"name": "Hybrid RAG API", "docs": "/docs", "health": "/health"}

    @app.get("/health", tags=["meta"])
    def health(svc: RAGService = Depends(service)):
        return svc.health()

    # Sync `def` endpoints run in FastAPI's threadpool, so blocking LangChain
    # calls (embedding, Qdrant, SQLite, LLM HTTP) never stall the event loop.
    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(body: ChatRequest, svc: RAGService = Depends(service)):
        session_id = body.session_id or uuid.uuid4().hex
        try:
            result = svc.ask(body.question, session_id=session_id,
                             route_override=body.route, retrieval_mode=body.retrieval_mode)
        except NotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - map LLM outages to 503, not 500
            mapped = _llm_http_error(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return ChatResponse.from_result(result, session_id)

    @app.post("/chat/stream", tags=["chat"])
    def chat_stream(body: ChatRequest, svc: RAGService = Depends(service)):
        """Server-Sent Events: `meta` (route + sources), `token`*, then `done` (full response)."""
        if svc.pipeline is None:
            raise HTTPException(status_code=503, detail="No data ingested yet.")
        session_id = body.session_id or uuid.uuid4().hex

        def sse(event: str, data) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def events() -> Iterator[str]:
            try:
                for kind, payload in svc.stream(body.question, session_id=session_id,
                                                route_override=body.route, retrieval_mode=body.retrieval_mode):
                    if kind == "token":
                        yield sse("token", {"text": payload})
                    else:
                        yield sse(kind, ChatResponse.from_result(payload, session_id).model_dump())
            except Exception as exc:  # noqa: BLE001 - headers are already sent; report in-band
                mapped = _llm_http_error(exc)
                if mapped is not None:
                    log.warning("Streaming aborted: %s", mapped.detail)
                    yield sse("error", {"detail": mapped.detail})
                else:
                    log.exception("Streaming failed")
                    yield sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/sessions/{session_id}", response_model=HistoryResponse, tags=["sessions"])
    def get_session(session_id: str, svc: RAGService = Depends(service)):
        messages = svc.sessions.recent(session_id, max_turns=10_000)
        return HistoryResponse(session_id=session_id, messages=[
            HistoryMessage(role="user" if m.type == "human" else "assistant", content=str(m.content))
            for m in messages
        ])

    @app.delete("/sessions/{session_id}", tags=["sessions"])
    def delete_session(session_id: str, svc: RAGService = Depends(service)):
        return {"session_id": session_id, "deleted": svc.sessions.clear(session_id)}

    @app.post("/admin/ingest", response_model=IngestResponse, tags=["admin"],
              dependencies=[Depends(require_admin)])
    def ingest(body: IngestRequest | None = None, svc: RAGService = Depends(service)):
        """Rebuild Qdrant and the graph backend from data/. Chat requests wait until it finishes."""
        stats = svc.ingest(recreate=(body or IngestRequest()).recreate)
        return IngestResponse(**stats.as_dict())

    @app.get("/admin/schema", tags=["admin"], dependencies=[Depends(require_admin)])
    def schema(svc: RAGService = Depends(service)):
        """The exact schema text given to the text-to-query LLM - useful when queries go wrong."""
        if svc.graph_store is None:
            raise HTTPException(status_code=503, detail=f"Graph backend unavailable: {svc.graph_error}")
        return {"backend": svc.settings.graph_backend, "language": svc.graph_store.language,
                "schema": svc.graph_store.schema_description(), "examples": svc.graph_store.examples()}

    return app


app = create_app()
