"""Shared types for the structured-data backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GraphResult:
    """Outcome of one natural-language -> query -> rows round trip."""

    language: str  # "sql" | "cypher"
    query: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 1
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_markdown(self, max_rows: int = 25) -> str:
        """Render rows as a Markdown table - a format LLMs read reliably."""
        if self.error:
            return f"(query failed: {self.error})"
        if not self.rows:
            return "(query succeeded but returned no rows)"
        cols = self.columns or list(self.rows[0])
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for row in self.rows[:max_rows]:
            lines.append("| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + " |")
        if len(self.rows) > max_rows:
            lines.append(f"\n({len(self.rows)} rows total, first {max_rows} shown)")
        return "\n".join(lines)


class GraphStore(Protocol):
    """What every backend must provide.

    This is the seam that makes the backend switchable: the query engine,
    router, and pipeline depend only on this protocol, never on SQLite or
    Neo4j directly.
    """

    language: str

    def schema_description(self) -> str:
        """Schema as text, injected into the text-to-query prompt."""
        ...

    def validate(self, query: str) -> str | None:
        """None if the query is safe to run, otherwise the rejection reason."""
        ...

    def run(self, query: str, limit: int = 50) -> tuple[list[str], list[dict[str, Any]]]:
        ...

    def examples(self) -> list[tuple[str, str]]:
        """(question, query) few-shot pairs synthesised from the real schema."""
        ...

    def table_names(self) -> list[str]:
        ...

    def vocabulary(self) -> set[str]:
        """Word tokens of table and column names, used by the router heuristic."""
        ...

    def close(self) -> None:
        ...
