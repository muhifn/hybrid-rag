from __future__ import annotations

from scripts.evaluate import (
    eval_end_to_end, eval_retrieval, eval_router, eval_text_to_query, markdown_report, plot_latency,
    plot_retrieval,
)


def test_full_evaluation_runs_and_writes_figures(service, gold, tmp_path):
    report = {
        "created": "test", "config": service.settings.public_dict(),
        "retrieval": eval_retrieval(service.retrievers, gold, service.settings.top_k),
        "router": eval_router(service, gold),
        "text_to_query": eval_text_to_query(service, gold),
        "end_to_end": eval_end_to_end(service, gold, judge=True),
    }

    r = report["retrieval"]
    assert r["cases"] == 10 and set(r["aggregate"]) == {"dense", "bm25", "hybrid"}
    assert all(0.0 <= v <= 1.0 for scores in r["aggregate"].values() for v in scores.values())

    t = report["text_to_query"]
    assert t["cases"] == 7 and t["execution_accuracy"] == 1.0  # fake LLM echoes the reference SQL

    e = report["end_to_end"]
    assert e["faithfulness_mean"] == 5 and e["relevance_mean"] == 4
    # Per-stage means are over all questions, so they sum to the mean total.
    assert abs(sum(e["stage_mean"].values()) - e["latency_mean"]) < 1e-6

    md = markdown_report(report)
    assert "Table 1. Retrieval" in md and "Table 3. Text-to-SQL" in md

    plot_retrieval(r, tmp_path / "f1.png")
    plot_latency(e, tmp_path / "f2.png")
    assert (tmp_path / "f1.png").stat().st_size > 10_000
    assert (tmp_path / "f2.png").stat().st_size > 5_000


def test_broken_reference_query_is_reported_not_fatal(service):
    broken = [{"question": "x", "route": "GRAPH", "relevant": [], "ref_sql": "SELECT nope FROM nowhere"}]
    result = eval_text_to_query(service, broken)
    assert result["detail"][0]["verdict"] == "REFERENCE_ERROR"
