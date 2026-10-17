"""Deterministic stand-ins for the chat model and the embedding model.

`FakeChatModel` is a real LangChain `BaseChatModel`, so tests run the actual
prompt | llm | parser chains, `with_structured_output`, and streaming - only
the text generation is scripted.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

# Canned SQL for questions the tests ask; anything else gets DEFAULT_SQL.
SQL = {
    "total anggaran proyek untuk setiap unit":
        "SELECT u.nama_unit, SUM(p.anggaran) AS total FROM proyek p "
        "JOIN unit u ON p.unit_id = u.unit_id GROUP BY u.nama_unit ORDER BY total DESC",
    "berapa proyek yang statusnya berjalan":
        "SELECT COUNT(*) AS jumlah FROM proyek WHERE status = 'berjalan'",
    "wajib menggunakan tender terbuka":
        "SELECT nama_proyek, anggaran FROM proyek WHERE anggaran > 200000000 ORDER BY anggaran DESC",
}
DEFAULT_SQL = "SELECT nama_proyek, anggaran FROM proyek ORDER BY anggaran DESC LIMIT 5"


def _system(messages: list[BaseMessage]) -> str:
    return next((str(m.content) for m in messages if m.type == "system"), "")


def _last_human(messages: list[BaseMessage]) -> str:
    return next((str(m.content) for m in reversed(messages) if m.type == "human"), "")


def scripted_reply(messages: list[BaseMessage], gold: list[dict] | None = None) -> str:
    system, human = _system(messages), _last_human(messages)
    low = human.lower()

    if system.startswith("You translate questions into"):
        if "self_correct_me" in low and "failed" not in low:
            return "SELECT * FROM tabel_tidak_ada"  # first attempt fails on purpose
        for key, sql in SQL.items():
            if key in low:
                return f"```sql\n{sql}\n```"  # fenced, to exercise the cleaner
        for item in gold or []:
            if item["question"].lower() in low and item.get("ref_sql"):
                return item["ref_sql"]
        return DEFAULT_SQL

    if system.startswith("You classify questions"):
        if "tender" in low or "pengadaan langsung" in low:
            route = "HYBRID"
        elif any(w in low for w in ("total", "ada berapa", "berapa jumlah", "tertinggi", "paling besar", "setiap unit")):
            route = "GRAPH"
        else:
            route = "VECTOR"
        return json.dumps({"route": route, "reason": "scripted"})

    if system.startswith("Rewrite the user's follow-up"):
        follow_up = re.search(r"Follow-up question: (.*?)\n", human)
        q = follow_up.group(1) if follow_up else human
        return "Apa yang terjadi jika cuti tahunan tidak digunakan?" if "tidak dipakai" in q else q

    if system.startswith("You answer questions"):
        cites = []
        if "[1]" in human:
            cites.append("[1]")
        if "[TABLE]" in human:
            cites.append("[TABLE]")
        return "Jawaban berdasarkan sumber " + " ".join(cites) + "."

    if system.startswith("You grade"):
        return json.dumps({"faithfulness": 5, "relevance": 4, "note": "scripted"})

    return "OK"


class FakeChatModel(BaseChatModel):
    responder: Callable[[list[BaseMessage]], str] = scripted_reply
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.responder(messages)))])

    def with_structured_output(self, schema, **kwargs: Any):
        def parse(prompt_value):
            self.calls += 1
            return schema.model_validate_json(self.responder(prompt_value.to_messages()))

        return RunnableLambda(parse)


class GarbageChatModel(FakeChatModel):
    """Never returns a usable route - forces the router's heuristic fallback."""

    responder: Callable[[list[BaseMessage]], str] = lambda messages: "I think maybe both? not sure"

    def with_structured_output(self, schema, **kwargs: Any):
        raise NotImplementedError("no structured output")


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashing: real lexical similarity under cosine, no model download."""

    def __init__(self, size: int = 256):
        self.size = size

    def _vector(self, text: str) -> list[float]:
        buckets = [0.0] * self.size
        for word in re.findall(r"\w+", text.lower()):
            buckets[zlib.crc32(word.encode()) % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
        return [v / norm for v in buckets]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
