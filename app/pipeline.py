"""The hybrid RAG flow: condense -> route -> retrieve (vector and/or graph) -> synthesize."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.graph.base import GraphResult
from app.graph.query_engine import GraphQueryEngine
from app.memory import SessionStore
from app.llm import text_parser
from app.prompts import ANSWER_PROMPT, CONDENSE_PROMPT, NOT_FOUND, build_context
from app.router import QueryRouter, RouteDecision
from app.text_utils import filter_reasoning_stream, strip_reasoning
from app.vectorstore import RetrieverSet

log = logging.getLogger(__name__)

EMPTY_ANSWER = ("(The model returned no answer text - its reasoning may have used up LLM_MAX_TOKENS. "
                "Increase LLM_MAX_TOKENS or turn off thinking for this model in LM Studio.)")


@dataclass
class RAGResult:
    question: str
    standalone_question: str
    answer: str
    route: str
    route_method: str
    route_reason: str
    documents: list[Document] = field(default_factory=list)
    graph: GraphResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    context: str = ""


@dataclass
class _Prepared:
    standalone: str
    decision: RouteDecision
    documents: list[Document]
    graph: GraphResult | None
    context: str
    timings: dict[str, float]


class HybridRAGPipeline:
    def __init__(
        self,
        llm: BaseChatModel,
        router: QueryRouter,
        retrievers: RetrieverSet,
        graph_engine: GraphQueryEngine | None,
        sessions: SessionStore,
        max_history_turns: int = 6,
        vector_lock: threading.Lock | None = None,
    ):
        self.router = router
        self.retrievers = retrievers
        self.graph_engine = graph_engine
        self.sessions = sessions
        self.max_history_turns = max_history_turns
        # Embedded Qdrant is not built for concurrent access; serialise searches.
        self._vector_lock = vector_lock or contextlib.nullcontext()
        self._condense = CONDENSE_PROMPT | llm.bind(temperature=0.0) | text_parser()
        self._answer = ANSWER_PROMPT | llm | StrOutputParser()

    # --------------------------------------------------------------- stages
    def condense(self, question: str, session_id: str | None) -> str:
        """Rewrite a follow-up into a standalone question (history-aware retrieval).

        "Kalau tidak dipakai bagaimana?" cannot be retrieved as-is; after
        rewriting it becomes "Apa yang terjadi jika cuti tahunan tidak digunakan?".
        """
        history = self.sessions.recent(session_id, self.max_history_turns) if session_id else []
        if not history:
            return question  # first turn: nothing to resolve, save an LLM call
        rewritten = self._condense.invoke({"history": history, "question": question}).strip().strip('"')
        # Small models sometimes answer instead of rewriting; detect and discard that.
        if not rewritten or len(rewritten) < 8 or len(rewritten) > 4 * len(question) + 120:
            return question
        return rewritten

    def _retrieve(self, query: str, mode: str) -> list[Document]:
        with self._vector_lock:
            return self.retrievers.retrieve(query, mode=mode)

    def _prepare(self, question: str, session_id: str | None, route_override: str | None,
                 retrieval_mode: str) -> _Prepared:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        standalone = self.condense(question, session_id)
        timings["condense"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if route_override:
            decision = RouteDecision(route=route_override.upper(), reason="set by caller", method="manual")
        else:
            decision = self.router.classify(standalone)
        if decision.route in ("GRAPH", "HYBRID") and self.graph_engine is None:
            decision = RouteDecision(route="VECTOR", reason="graph backend unavailable", method=decision.method)
        timings["router"] = time.perf_counter() - t0

        documents: list[Document] = []
        graph: GraphResult | None = None

        if decision.route in ("VECTOR", "HYBRID"):
            t0 = time.perf_counter()
            documents = self._retrieve(standalone, retrieval_mode)
            timings["vector"] = time.perf_counter() - t0

        if decision.route in ("GRAPH", "HYBRID"):
            t0 = time.perf_counter()
            graph = self.graph_engine.ask(standalone)
            timings["graph"] = time.perf_counter() - t0
            # Recovery: if a pure GRAPH question fails, still try the documents
            # so the user gets an answer instead of an error.
            if decision.route == "GRAPH" and not graph.ok:
                t0 = time.perf_counter()
                documents = self._retrieve(standalone, retrieval_mode)
                timings["vector_fallback"] = time.perf_counter() - t0

        context = build_context(documents, graph)
        return _Prepared(standalone, decision, documents, graph, context, timings)

    def _result(self, question: str, p: _Prepared, answer: str) -> RAGResult:
        p.timings["total"] = sum(v for k, v in p.timings.items() if k != "total")
        return RAGResult(
            question=question, standalone_question=p.standalone, answer=answer,
            route=p.decision.route, route_method=p.decision.method, route_reason=p.decision.reason,
            documents=p.documents, graph=p.graph, timings=p.timings, context=p.context,
        )

    @staticmethod
    def _has_context(p: _Prepared) -> bool:
        return bool(p.documents) or (p.graph is not None and p.graph.ok)

    # --------------------------------------------------------------- public
    def ask(self, question: str, session_id: str | None = None, route_override: str | None = None,
            retrieval_mode: str = "hybrid") -> RAGResult:
        p = self._prepare(question, session_id, route_override, retrieval_mode)
        t0 = time.perf_counter()
        if self._has_context(p):
            answer = strip_reasoning(self._answer.invoke({"context": p.context, "question": p.standalone}))
            answer = answer or EMPTY_ANSWER
        else:
            answer = NOT_FOUND
        p.timings["synthesis"] = time.perf_counter() - t0
        if session_id:
            self.sessions.append(session_id, question, answer)
        return self._result(question, p, answer)

    def stream(self, question: str, session_id: str | None = None, route_override: str | None = None,
               retrieval_mode: str = "hybrid") -> Iterator[tuple[str, Any]]:
        """Yield ("meta", RAGResult-without-answer), ("token", str)..., ("done", RAGResult)."""
        p = self._prepare(question, session_id, route_override, retrieval_mode)
        yield "meta", self._result(question, p, "")

        t0 = time.perf_counter()
        parts: list[str] = []
        if self._has_context(p):
            raw = self._answer.stream({"context": p.context, "question": p.standalone})
            for token in filter_reasoning_stream(raw):
                parts.append(token)
                yield "token", token
            if not "".join(parts).strip():
                parts = [EMPTY_ANSWER]
                yield "token", EMPTY_ANSWER
        else:
            parts.append(NOT_FOUND)
            yield "token", NOT_FOUND
        answer = "".join(parts)
        p.timings["synthesis"] = time.perf_counter() - t0

        if session_id:
            self.sessions.append(session_id, question, answer)
        yield "done", self._result(question, p, answer)
