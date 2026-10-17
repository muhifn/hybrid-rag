"""Evaluate the system - material for the thesis Results chapter.

    python -m scripts.evaluate                    # full run (needs LM Studio)
    python -m scripts.evaluate --retrieval-only   # dense vs bm25 vs hybrid; no LLM needed
    python -m scripts.evaluate --no-judge         # skip the LLM-as-judge step

Writes to results/:
    eval_<timestamp>.json    every number plus the configuration (appendix)
    eval_<timestamp>.md      summary tables, ready to paste into the thesis
    figure1_retrieval.png    retrieval metrics per mode
    figure2_latency.png      mean time per pipeline stage

| # | Measures                 | Metric                              |
|---|--------------------------|-------------------------------------|
| 1 | retrieval quality        | Hit@k, Recall@k, MRR, nDCG@k        |
| 2 | router correctness       | accuracy, confusion matrix          |
| 3 | text-to-query            | execution accuracy, first-try rate  |
| 4 | end-to-end               | latency per stage, judge scores     |

Limitations to state in the Discussion chapter: the bundled gold set has only
15 questions over sample data, and the judge is the same model that wrote the
answers (self-preference bias, Zheng et al., 2023). Treat judge scores as a
comparison between configurations, not as absolute quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.config import PROJECT_ROOT, get_settings
from app.llm import text_parser
from app.metrics import canonical_rows, dedupe, hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from app.router import ROUTES
from app.text_utils import extract_json
from app.vectorstore import RETRIEVAL_MODES, RetrieverSet

log = logging.getLogger("evaluate")

METRICS = ("Hit@k", "Recall@k", "MRR", "nDCG@k")

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade answers from a question-answering system, using ONLY the given context.\n"
     "faithfulness (1-5): how fully the answer is supported by the context. "
     "5 = every claim supported; 1 = many claims absent from the context.\n"
     "relevance (1-5): how directly the answer addresses the question. 5 = on point; 1 = off topic.\n"
     "If the answer says the information was not found AND the context indeed lacks it, "
     "that is faithfulness 5 (a correct refusal).\n"
     'Reply with JSON only: {{"faithfulness": 1-5, "relevance": 1-5, "note": "one sentence"}}'),
    ("human", "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])


# --------------------------------------------------------------------- evals
def eval_retrieval(retrievers: RetrieverSet, gold: list[dict], k: int) -> dict[str, Any]:
    cases = [g for g in gold if g.get("relevant")]
    aggregate: dict[str, dict[str, float]] = {}
    detail: list[dict] = []
    for mode in RETRIEVAL_MODES:
        totals = dict.fromkeys(METRICS, 0.0)
        for case in cases:
            ranked = dedupe([d.metadata["source"] for d in retrievers.retrieve(case["question"], mode)])
            rel = case["relevant"]
            scores = {"Hit@k": hit_at_k(ranked, rel, k), "Recall@k": recall_at_k(ranked, rel, k),
                      "MRR": reciprocal_rank(ranked, rel), "nDCG@k": ndcg_at_k(ranked, rel, k)}
            for m in METRICS:
                totals[m] += scores[m]
            detail.append({"mode": mode, "question": case["question"], "retrieved": ranked,
                           **{m: round(v, 4) for m, v in scores.items()}})
        aggregate[mode] = {m: totals[m] / len(cases) for m in METRICS} if cases else {}
    return {"k": k, "cases": len(cases), "aggregate": aggregate, "detail": detail}


def eval_router(service, gold: list[dict]) -> dict[str, Any]:
    confusion: Counter = Counter()
    methods: Counter = Counter()
    rows, llm_ok, heur_ok = [], 0, 0
    for case in gold:
        decision = service.router.classify(case["question"])
        heuristic = service.router.classify(case["question"], use_llm=False)
        llm_ok += decision.route == case["route"]
        heur_ok += heuristic.route == case["route"]
        methods[decision.method] += 1
        confusion[(case["route"], decision.route)] += 1
        rows.append({"question": case["question"], "expected": case["route"], "llm": decision.route,
                     "heuristic": heuristic.route, "method": decision.method})
    n = len(gold)
    return {
        "cases": n, "accuracy_llm": llm_ok / n, "accuracy_heuristic": heur_ok / n,
        "decided_by": dict(methods),
        "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
        "detail": rows,
    }


def eval_text_to_query(service, gold: list[dict]) -> dict[str, Any]:
    store, engine = service.graph_store, service.graph_engine
    key = "ref_sql" if store.language == "sql" else "ref_cypher"
    cases = [g for g in gold if g.get(key)]
    rows, correct, attempts = [], 0, Counter()
    for case in cases:
        try:
            _, reference = store.run(case[key], limit=500)
        except Exception as exc:  # noqa: BLE001 - a broken gold query must be reported, not crash the run
            rows.append({"question": case["question"], "verdict": "REFERENCE_ERROR", "attempts": 0,
                         "generated": None, "reference": case[key], "error": f"{type(exc).__name__}: {exc}"})
            log.error("Reference %s for %r is broken: %s", key, case["question"], exc)
            continue
        produced = engine.ask(case["question"])
        if not produced.ok:
            verdict = "FAILED"
        elif canonical_rows(produced.rows) == canonical_rows(reference):
            verdict = "CORRECT"
        else:
            verdict = "WRONG"
        correct += verdict == "CORRECT"
        attempts[produced.attempts if produced.ok else "failed"] += 1
        rows.append({"question": case["question"], "verdict": verdict, "attempts": produced.attempts,
                     "generated": produced.query, "reference": case[key], "error": produced.error})
        log.info("[%s] %s", verdict, case["question"])
    n = len(cases) or 1
    first_try = sum(1 for r in rows if r["verdict"] == "CORRECT" and r["attempts"] == 1)
    return {
        "language": store.language, "cases": len(cases),
        "execution_accuracy": correct / n, "first_try_accuracy": first_try / n,
        "attempt_histogram": {str(k): v for k, v in attempts.items()}, "detail": rows,
    }


def eval_end_to_end(service, gold: list[dict], judge: bool) -> dict[str, Any]:
    judge_chain = JUDGE_PROMPT | service.llm.bind(temperature=0.0) | text_parser()
    stages: dict[str, list[float]] = defaultdict(list)
    rows, faith, relev = [], [], []
    for n, case in enumerate(gold, start=1):
        result = service.ask(case["question"])  # no session: each question is independent
        for stage, seconds in result.timings.items():
            stages[stage].append(seconds)
        f = r = None
        if judge:
            verdict = extract_json(judge_chain.invoke({
                "question": case["question"], "context": result.context[:6000], "answer": result.answer,
            }))
            if isinstance(verdict, dict):
                try:
                    f = max(1, min(5, int(verdict["faithfulness"])))
                    r = max(1, min(5, int(verdict["relevance"])))
                    faith.append(f)
                    relev.append(r)
                except (KeyError, TypeError, ValueError):
                    pass
        rows.append({"question": case["question"], "expected_route": case["route"], "route": result.route,
                     "seconds": round(result.timings["total"], 3), "faithfulness": f, "relevance": r,
                     "answer": result.answer})
        log.info("[%2d/%d] %-6s %6.2fs  %s", n, len(gold), result.route, result.timings["total"], case["question"])
    totals = stages.pop("total", [0.0])
    return {
        "latency_mean": statistics.mean(totals), "latency_median": statistics.median(totals),
        "latency_max": max(totals),
        # Mean over ALL questions (a stage a question skipped counts as 0s), so
        # the stages add up to the mean total.
        "stage_mean": {s: sum(v) / len(gold) for s, v in stages.items()},
        "faithfulness_mean": statistics.mean(faith) if faith else None,
        "relevance_mean": statistics.mean(relev) if relev else None,
        "judge_note": "Judge is the same model that wrote the answers; prone to self-preference bias.",
        "detail": rows,
    }


# ------------------------------------------------------------------ outputs
# Categorical slots in fixed order; validated for colour-vision deficiency on
# the light surface. Low-contrast slots are relieved by direct value labels and
# by the tables in the Markdown report.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _pyplot():
    """Headless matplotlib with the shared chart style applied."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def _style(ax, horizontal: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x" if horizontal else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_retrieval(retrieval: dict, path: Path) -> None:
    from matplotlib.ticker import MultipleLocator

    plt = _pyplot()

    modes, k = list(retrieval["aggregate"]), retrieval["k"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.78 / len(modes)
    for slot, mode in enumerate(modes):
        xs = [i - 0.39 + width * (slot + 0.5) for i in range(len(METRICS))]
        values = [retrieval["aggregate"][mode][m] for m in METRICS]
        # Surface-coloured 2px edge = visible gap between adjacent bars.
        bars = ax.bar(xs, values, width=width * 0.94, color=SERIES[slot], edgecolor=SURFACE,
                      linewidth=2, label=mode, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK2, zorder=4)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m.replace("@k", f"@{k}") for m in METRICS], color=INK2, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_ylabel("score (0-1)", color=INK2, fontsize=10)
    ax.set_title(f"Figure 1. Retrieval performance by mode (n={retrieval['cases']} questions)",
                 color=INK, fontsize=12, fontweight="600", loc="left", pad=14)
    ax.legend(frameon=False, ncols=len(modes), loc="upper right", fontsize=9, labelcolor=INK2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latency(e2e: dict, path: Path) -> None:
    plt = _pyplot()
    stage_mean = dict(e2e["stage_mean"])
    # Four stages -> the four validated slots. Condense is ~0 here (evaluation
    # uses no session) and the vector fallback is folded into vector.
    stage_mean["vector"] = stage_mean.get("vector", 0.0) + stage_mean.pop("vector_fallback", 0.0)
    stage_mean.pop("condense", None)
    labels = {"router": "Router", "vector": "Vector retrieval", "graph": "Graph query", "synthesis": "LLM synthesis"}
    order = [s for s in labels if stage_mean.get(s, 0) > 0]
    total = sum(stage_mean[s] for s in order) or 1.0

    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    # Fixed margins instead of tight_layout: the bottom strip is reserved for a
    # figure-level legend, so legend, axis label, and bar can never collide.
    fig.subplots_adjust(left=0.03, right=0.98, top=0.72, bottom=0.40)
    left = 0.0
    for slot, stage in enumerate(order):
        seconds = stage_mean[stage]
        ax.barh([0], [seconds], left=left, height=0.6, color=SERIES[slot], edgecolor=SURFACE, linewidth=2,
                label=f"{labels[stage]} ({seconds:.2f}s)", zorder=3)
        if seconds > total * 0.12:  # label inside only when it fits
            # Dark ink: >= 4.8:1 on all four fills; white fails on aqua and yellow.
            ax.text(left + seconds / 2, 0, f"{seconds:.2f}s", ha="center", va="center",
                    fontsize=9, color=INK, fontweight="600", zorder=4)
        left += seconds
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("seconds (mean per question)", color=INK2, fontsize=10)
    ax.set_title(f"Figure 2. Latency by pipeline stage (total {left:.2f}s)",
                 color=INK, fontsize=12, fontweight="600", loc="left", pad=12)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncols=len(order), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), fontsize=9, labelcolor=INK2)
    _style(ax, horizontal=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_report(report: dict) -> str:
    out = [f"# Evaluation report - {report['created']}", ""]
    cfg = report["config"]
    out += ["## Configuration", "", "| setting | value |", "|---|---|"]
    out += [f"| {k} | {cfg[k]} |" for k in ("llm_model", "embedding_model", "chunk_size", "chunk_overlap",
                                             "top_k", "bm25_weight", "graph_backend") if k in cfg]
    r = report["retrieval"]
    out += ["", f"## Table 1. Retrieval (k={r['k']}, n={r['cases']})", "",
            "| mode | " + " | ".join(m.replace("@k", f"@{r['k']}") for m in METRICS) + " |",
            "|---" * (len(METRICS) + 1) + "|"]
    for mode, scores in r["aggregate"].items():
        out.append(f"| {mode} | " + " | ".join(f"{scores[m]:.3f}" for m in METRICS) + " |")
    if "router" in report:
        ro = report["router"]
        out += ["", f"## Table 2. Router (n={ro['cases']})", "",
                f"- LLM router accuracy: **{ro['accuracy_llm']:.1%}**",
                f"- Heuristic-only accuracy: **{ro['accuracy_heuristic']:.1%}**",
                f"- Decided by: {ro['decided_by']}", "",
                "| expected \\ predicted | " + " | ".join(ROUTES) + " |", "|---" * (len(ROUTES) + 1) + "|"]
        for exp in ROUTES:
            out.append(f"| {exp} | " + " | ".join(str(ro["confusion"].get(f"{exp}->{p}", 0)) for p in ROUTES) + " |")
    if "text_to_query" in report:
        t = report["text_to_query"]
        out += ["", f"## Table 3. Text-to-{t['language'].upper()} (n={t['cases']})", "",
                f"- Execution accuracy: **{t['execution_accuracy']:.1%}**",
                f"- Correct on first attempt: **{t['first_try_accuracy']:.1%}**",
                f"- Attempts: {t['attempt_histogram']}",
                "", "The gap between the two is the measured contribution of the self-correction loop."]
    if "end_to_end" in report:
        e = report["end_to_end"]
        out += ["", "## Table 4. End-to-end", "",
                f"- Latency mean / median / max: {e['latency_mean']:.2f}s / {e['latency_median']:.2f}s / "
                f"{e['latency_max']:.2f}s", "", "| stage | mean seconds |", "|---|---|"]
        out += [f"| {s} | {v:.3f} |" for s, v in sorted(e["stage_mean"].items(), key=lambda kv: -kv[1])]
        if e["faithfulness_mean"] is not None:
            out += ["", f"- Judge faithfulness: {e['faithfulness_mean']:.2f}/5, "
                        f"relevance: {e['relevance_mean']:.2f}/5", f"- Note: {e['judge_note']}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, default=PROJECT_ROOT / "eval" / "gold_set.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--retrieval-only", action="store_true", help="skip everything that needs the LLM")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "app.graph.query_engine", "app.vectorstore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = get_settings()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report: dict[str, Any] = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "gold_set": str(args.gold),
                              "config": settings.public_dict()}

    if args.retrieval_only:
        from app.llm import build_embeddings, embedding_dimension
        from app.vectorstore import VectorIndex, build_retrievers

        embeddings = build_embeddings(settings)
        index = VectorIndex(settings, embeddings, embedding_dimension(embeddings))
        try:
            if index.count() == 0:
                print("Vector store is empty. Run: python -m scripts.ingest", file=sys.stderr)
                return 1
            report["retrieval"] = eval_retrieval(build_retrievers(index, settings), gold, settings.top_k)
        finally:
            index.close()
    else:
        from app.service import RAGService

        service = RAGService(settings)
        try:
            if service.pipeline is None:
                print("No data ingested. Run: python -m scripts.ingest", file=sys.stderr)
                return 1
            report["config"]["llm_model"] = service.model_name
            log.info("1/4 retrieval")
            report["retrieval"] = eval_retrieval(service.retrievers, gold, settings.top_k)
            log.info("2/4 router")
            report["router"] = eval_router(service, gold)
            if service.graph_store is not None:
                log.info("3/4 text-to-query")
                report["text_to_query"] = eval_text_to_query(service, gold)
            log.info("4/4 end-to-end")
            report["end_to_end"] = eval_end_to_end(service, gold, judge=not args.no_judge)
        finally:
            service.close()

    json_path = args.out / f"eval_{stamp}.json"
    md_path = args.out / f"eval_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    plot_retrieval(report["retrieval"], args.out / "figure1_retrieval.png")
    if "end_to_end" in report:
        plot_latency(report["end_to_end"], args.out / "figure2_latency.png")

    print(markdown_report(report))
    print(f"Saved: {json_path}\n       {md_path}\n       {args.out / 'figure1_retrieval.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
