from __future__ import annotations

import pytest

from app.metrics import canonical_rows, dedupe, hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_dedupe_keeps_first_occurrence():
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


@pytest.mark.parametrize("ranked,relevant,k,expected", [
    (["a", "b"], ["b"], 2, 1.0),
    (["a", "b"], ["c"], 2, 0.0),
    (["a", "b", "c"], ["c"], 2, 0.0),  # relevant item outside k
])
def test_hit_at_k(ranked, relevant, k, expected):
    assert hit_at_k(ranked, relevant, k) == expected


def test_recall_at_k():
    assert recall_at_k(["a"], ["a", "z"], 3) == 0.5
    assert recall_at_k(["a"], [], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_ndcg():
    assert ndcg_at_k(["a", "b"], ["a"], 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], ["a"], 2) == pytest.approx(1 / 1.5849625, rel=1e-6)


def test_canonical_rows_ignore_order_types_and_column_names():
    assert canonical_rows([{"x": 1}, {"x": 2}]) == canonical_rows([{"y": 2.0}, {"y": "1"}])
    assert canonical_rows([{"a": "Jakarta", "b": 3}]) == canonical_rows([{"b": 3, "a": "jakarta"}])
    assert canonical_rows([{"x": 1}]) != canonical_rows([{"x": 2}])
