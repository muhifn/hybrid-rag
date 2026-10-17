"""Small text helpers shared by BM25, the SQL guard, and the router."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Iterator

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_reasoning(text: str) -> str:
    """Remove `<think>...</think>` blocks that reasoning models (Qwen3.x, DeepSeek-R1)
    may emit inline. LM Studio usually moves reasoning to a separate field, but
    when it does not, the tags would corrupt generated SQL, router JSON, and
    the displayed answer. An unterminated block (generation cut off by the
    token limit) is dropped entirely.
    """
    if not text or _THINK_OPEN not in text.lower():
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    start = cleaned.lower().find(_THINK_OPEN)
    if start != -1:
        cleaned = cleaned[:start]
    return cleaned.strip()


def filter_reasoning_stream(tokens: Iterable[str]) -> Iterator[str]:
    """Streaming counterpart of `strip_reasoning` for a leading think block.

    States: deciding (could the output still start with <think>?) ->
    thinking (swallow until </think>) -> trimming (drop leading whitespace) ->
    passthrough.
    """
    buffer, state = "", "deciding"
    for token in tokens:
        if state == "passthrough":
            yield token
            continue
        buffer += token
        if state == "deciding":
            head = buffer.lstrip().lower()
            if head.startswith(_THINK_OPEN):
                state = "thinking"
            elif _THINK_OPEN.startswith(head):
                continue  # may still turn into "<think>"
            else:
                state = "passthrough"
                yield buffer
                buffer = ""
                continue
        if state == "thinking":
            end = buffer.lower().find(_THINK_CLOSE)
            if end == -1:
                continue
            buffer = buffer[end + len(_THINK_CLOSE):]
            state = "trimming"
        # trimming: the answer usually starts after blank lines
        buffer = buffer.lstrip()
        if buffer:
            state = "passthrough"
            yield buffer
            buffer = ""
    if state == "deciding" and buffer:
        yield buffer


def tokenize(text: str) -> list[str]:
    """Lower-cased, Unicode-aware word tokens.

    Used as BM25's `preprocess_func`. LangChain's default is `str.split()`,
    which is case-sensitive and keeps punctuation attached ("cuti," != "Cuti"),
    noticeably hurting recall on Indonesian text.
    """
    tokens: list[str] = []
    buffer: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buffer.append(ch)
        elif buffer:
            tokens.append("".join(buffer))
            buffer = []
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def extract_json(raw: str) -> Any | None:
    """Recover the first JSON value from LLM output that may wrap it in prose or fences."""
    if not raw:
        return None
    text = strip_reasoning(raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate[:1] in "{[":
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


def strip_code_fences(raw: str, labels: tuple[str, ...] = ("sql", "cypher")) -> str:
    """Remove ``` fences and leading language labels models like to add around queries."""
    text = strip_reasoning(raw or "").strip()
    if "```" in text:
        for segment in text.split("```"):
            candidate = segment.strip()
            for label in labels:
                if candidate.lower().startswith(label):
                    candidate = candidate[len(label) :].strip()
            if candidate:
                text = candidate
                break
    for label in labels:
        prefix = f"{label}:"
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    return text.strip().rstrip(";").strip()
