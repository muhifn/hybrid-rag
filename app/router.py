"""Route each question to VECTOR, GRAPH, or HYBRID retrieval.

| Route  | When                                       | Example                                       |
|--------|--------------------------------------------|-----------------------------------------------|
| VECTOR | answer lives in document prose             | "Bagaimana prosedur pengajuan cuti?"          |
| GRAPH  | needs aggregation / exact numbers / joins  | "Berapa total anggaran per unit?"             |
| HYBRID | needs a rule from documents AND table data | "Proyek mana yang wajib tender terbuka?"      |

Two layers, because 7B local models occasionally return an invalid label:
  1. LLM classification with JSON-schema constrained decoding
     (`with_structured_output`; LM Studio enforces the schema with a grammar)
  2. a keyword heuristic (Indonesian + English cues, plus schema-term matches)
     used whenever layer 1 fails

`RouteDecision.method` records which layer decided, so the evaluation can
report how often the fallback fires and compare both layers' accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app.llm import text_parser
from app.text_utils import extract_json, tokenize

log = logging.getLogger(__name__)

ROUTES = ("VECTOR", "GRAPH", "HYBRID")


class RouteChoice(BaseModel):
    """Structured-output schema for the router."""

    route: Literal["VECTOR", "GRAPH", "HYBRID"] = Field(description="Which data source answers the question")
    reason: str = Field(description="One short sentence explaining the choice")


@dataclass
class RouteDecision:
    route: str
    reason: str
    method: str  # "llm" | "heuristic" | "fallback" | "manual"
    scores: dict[str, int] = field(default_factory=dict)


SYSTEM = """You classify questions for a hybrid RAG system. Decide which data source is needed.

STRUCTURED DATA (database tables): {tables}

TEXT DOCUMENTS: {documents}

ROUTES:
  VECTOR - the answer is prose, an explanation, a rule, or a procedure from the text documents
  GRAPH  - needs exact numbers, aggregation (total/average/count), ranking, comparison,
           or relations between rows of the tables
  HYBRID - needs a rule from the documents AND data from the tables together

Questions are usually in Indonesian."""

# Injected as a prompt *value* via .partial(), so braces are literal - no {{ }} escaping.
JSON_INSTRUCTION = '\n\nReply with JSON only: {"route": "VECTOR|GRAPH|HYBRID", "reason": "one sentence"}'


class QueryRouter:
    GRAPH_CUES = (
        # Indonesian
        "berapa", "jumlah", "total", "rata-rata", "rata rata", "tertinggi", "terendah",
        "terbesar", "terkecil", "paling", "maksimum", "minimum", "urutkan", "peringkat",
        "daftar", "sebutkan", "hitung", "masing-masing", "setiap", "persen",
        "lebih dari", "kurang dari", "di atas", "di bawah", "siapa saja",
        # English
        "how many", "how much", "average", "highest", "lowest", "count", "sum", "list ",
    )
    VECTOR_CUES = (
        # Indonesian
        "bagaimana", "apa itu", "jelaskan", "mengapa", "kenapa", "prosedur", "cara",
        "syarat", "ketentuan", "kebijakan", "aturan", "panduan", "definisi", "maksud",
        "tujuan", "langkah", "boleh", "wajib", "dilarang", "sanksi", "berlaku",
        # English
        "how do", "how to", "what is", "explain", "why", "procedure", "policy", "rule",
    )

    def __init__(self, llm: BaseChatModel, table_names: list[str], schema_terms: set[str],
                 document_titles: list[str], use_structured_output: bool = True):
        self.schema_terms = schema_terms
        system = SYSTEM.format(
            tables=", ".join(table_names) or "(none)",
            documents="; ".join(document_titles[:15]) or "(none)",
        )
        self._structured = None
        if use_structured_output:
            try:
                prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{question}")])
                self._structured = prompt.partial(system=system) | llm.with_structured_output(
                    RouteChoice, method="json_schema"
                )
            except (NotImplementedError, ValueError, TypeError) as exc:
                log.warning("Structured output unavailable (%s); using JSON prompt", exc)
        json_prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{question}")])
        self._plain = json_prompt.partial(system=system + JSON_INSTRUCTION) | llm | text_parser()

    # ---------------------------------------------------------------- layers
    def _llm(self, question: str) -> RouteChoice | None:
        if self._structured is not None:
            try:
                choice = self._structured.invoke({"question": question})
                if isinstance(choice, RouteChoice):
                    return choice
            except Exception as exc:  # noqa: BLE001 - any failure falls through
                log.info("Structured routing failed (%s: %s); retrying with JSON prompt",
                         type(exc).__name__, str(exc)[:120])
        try:
            payload = extract_json(self._plain.invoke({"question": question}))
            if isinstance(payload, dict):
                payload["route"] = str(payload.get("route", "")).strip().upper()
                return RouteChoice.model_validate(payload)
        except (ValidationError, Exception) as exc:  # noqa: BLE001
            log.info("JSON routing failed: %s", str(exc)[:120])
        return None

    def heuristic(self, question: str) -> RouteDecision:
        low = question.lower()
        schema_hits = len(set(tokenize(low)) & self.schema_terms)
        graph = sum(cue in low for cue in self.GRAPH_CUES) + schema_hits
        vector = sum(cue in low for cue in self.VECTOR_CUES)
        scores = {"graph": graph, "vector": vector, "schema": schema_hits}
        if graph and vector:
            route, why = "HYBRID", "has both aggregation and procedure cues"
        elif graph > vector:
            route, why = "GRAPH", f"aggregation/schema cues ({graph})"
        elif vector > graph:
            route, why = "VECTOR", f"procedure/definition cues ({vector})"
        else:
            # VECTOR is the safe default: it never fails hard, at worst returns weak context.
            route, why = "VECTOR", "no strong cues; default to document search"
        return RouteDecision(route=route, reason=why, method="heuristic", scores=scores)

    def classify(self, question: str, use_llm: bool = True) -> RouteDecision:
        if not use_llm:
            return self.heuristic(question)
        choice = self._llm(question)
        if choice is not None:
            return RouteDecision(route=choice.route, reason=choice.reason, method="llm")
        decision = self.heuristic(question)
        decision.method = "fallback"
        decision.reason = f"LLM gave no valid route; heuristic: {decision.reason}"
        return decision
