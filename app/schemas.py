"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.pipeline import RAGResult

Route = Literal["VECTOR", "GRAPH", "HYBRID"]
RetrievalMode = Literal["dense", "bm25", "hybrid"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["Berapa hari cuti tahunan pegawai?"])
    session_id: str | None = Field(default=None, max_length=128,
                                   description="Reuse to keep conversation memory; omit for a new session")
    route: Route | None = Field(default=None, description="Force a route instead of using the router")
    retrieval_mode: RetrievalMode = "hybrid"


class SourceOut(BaseModel):
    rank: int
    source: str
    title: str
    page: int | None = None
    chunk_id: str | None = None
    snippet: str


class GraphOut(BaseModel):
    language: str
    query: str
    ok: bool
    attempts: int
    error: str | None = None
    columns: list[str]
    rows: list[dict[str, Any]]


class ChatResponse(BaseModel):
    session_id: str
    question: str
    standalone_question: str
    answer: str
    route: str
    route_method: str
    route_reason: str
    sources: list[SourceOut]
    graph: GraphOut | None
    timings: dict[str, float]

    @classmethod
    def from_result(cls, result: RAGResult, session_id: str) -> "ChatResponse":
        sources = [
            SourceOut(
                rank=n,
                source=d.metadata.get("source", "?"),
                title=d.metadata.get("title", "?"),
                page=d.metadata.get("page"),
                chunk_id=d.metadata.get("chunk_id"),
                snippet=" ".join(d.page_content.split())[:280],
            )
            for n, d in enumerate(result.documents, start=1)
        ]
        graph = None
        if result.graph is not None:
            g = result.graph
            graph = GraphOut(language=g.language, query=g.query, ok=g.ok, attempts=g.attempts,
                             error=g.error, columns=g.columns, rows=g.rows[:50])
        return cls(
            session_id=session_id, question=result.question,
            standalone_question=result.standalone_question, answer=result.answer,
            route=result.route, route_method=result.route_method, route_reason=result.route_reason,
            sources=sources, graph=graph,
            timings={k: round(v, 3) for k, v in result.timings.items()},
        )


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class IngestRequest(BaseModel):
    recreate: bool = Field(default=True, description="Drop and rebuild the vector collection")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    tables: int
    table_rows: int
    graph_backend: str
    seconds: float
