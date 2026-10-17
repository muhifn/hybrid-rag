"""Evaluation metrics (document-level retrieval metrics and result-set comparison).

Retrieval metrics are computed per DOCUMENT, not per chunk: the set of
relevant chunks changes whenever chunk_size changes, so chunk-level gold labels
would not survive the chunking experiments. Document labels stay valid across
configurations, which keeps comparisons fair.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def dedupe(items: Sequence[str]) -> list[str]:
    """Keep first occurrence: several chunks of one document count once, at their best rank."""
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def hit_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    return next((1.0 / i for i, d in enumerate(ranked, start=1) if d in rel), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


def _canonical_value(value: Any) -> str:
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        return f"{float(value):.4f}"  # 5, 5.0 and "5" compare equal
    except (TypeError, ValueError):
        return str(value).strip().casefold()


def canonical_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[str, ...]]:
    """Order-insensitive form of a result set, for execution accuracy.

    Ignores column names/order and row order: two differently written queries
    that return the same values are both correct (as in the Spider benchmark,
    Yu et al., 2018).
    """
    return sorted(tuple(sorted(_canonical_value(v) for v in row.values())) for row in rows)
