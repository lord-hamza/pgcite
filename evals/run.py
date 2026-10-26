import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evals import metrics
from pgcite.pipeline import RAGPipeline

DEFAULT_TOLERANCE = 0.02
ABLATION = [
    ("bm25", "bm25", False),
    ("vector", "vector", False),
    ("hybrid", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]


def load_golden(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(
    pipeline: RAGPipeline, golden: list[dict], k: int, mode: str, rerank: bool
) -> list[dict]:
    embedder = pipeline.embedder
    rows = []
    for item in golden:
        result = pipeline.ask(item["question"], k=k, mode=mode, rerank=rerank)
        docs = [hit.doc_id for hit in result.hits]
        contexts = {i: hit.content for i, hit in enumerate(result.hits, start=1)}
        answer = result.answer.text
        faithfulness = metrics.faithfulness(answer, contexts, embedder)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "model": result.answer.model,
                "recall": metrics.recall_at_k(docs, item["relevant_docs"], k),
                "rr": metrics.reciprocal_rank(docs, item["relevant_docs"]),
                "faithfulness": faithfulness,
                "relevance": metrics.answer_relevance(item["question"], answer, embedder),
                "similarity": metrics.answer_similarity(item["answer"], answer, embedder),
                "ms": result.timings_ms["total"],
                "cost": result.answer.cost_usd,
            }
        )
    return rows


def run_ablation(pipeline: RAGPipeline, golden: list[dict], k: int) -> list[tuple]:
    table = []
    for label, mode, rerank in ABLATION:
        recalls, ranks, latencies = [], [], []
        for item in golden:
            result = pipeline.search(item["question"], k=k, mode=mode, rerank=rerank)
            docs = [hit.doc_id for hit in result.hits]
            recalls.append(metrics.recall_at_k(docs, item["relevant_docs"], k))
            ranks.append(metrics.reciprocal_rank(docs, item["relevant_docs"]))
            latencies.append(result.timings_ms["total"])
        table.append((label, np.mean(recalls), np.mean(ranks), np.median(latencies)))
    return table


def summarize(rows: list[dict], k: int) -> dict[str, float]:
    columns = [
        (f"recall@{k}", "recall"),
        ("mrr", "rr"),
        ("faithfulness", "faithfulness"),
        ("answer_relevance", "relevance"),
        ("answer_similarity", "similarity"),
    ]
    return {name: round(float(np.mean([row[key] for row in rows])), 3) for name, key in columns}


def gate(scores: dict[str, float], baseline: dict | None) -> tuple[list[tuple], bool]:
    tolerance = baseline.get("tolerance", DEFAULT_TOLERANCE) if baseline else DEFAULT_TOLERANCE
    reference = baseline["metrics"] if baseline else {}
    table = []
    for name, value in scores.items():
        base = reference.get(name)
        if base is None:
            status = "new"
        elif value < round(base - tolerance, 6):
            status = "FAIL"
        else:
            status = "ok"
        table.append((name, base, value, status))
    return table, any(status == "FAIL" for *_, status in table)


def render_summary(
    config: str, gate_table: list[tuple], rows: list[dict], ablation: list[tuple], k: int
) -> str:
    lines = [
        f"# Evaluation report\n\n{config}\n",
        "| metric | baseline | current | delta | status |",
        "|---|---:|---:|---:|---|",
    ]
    for name, base, value, status in gate_table:
        base_text, delta = ("-", "-") if base is None else (f"{base:.3f}", f"{value - base:+.3f}")
        lines.append(f"| {name} | {base_text} | {value:.3f} | {delta} | {status} |")
    latencies = [row["ms"] for row in rows]
    lines.append(
        f"\nLatency per question: p50 {np.percentile(latencies, 50):.1f} ms, "
        f"p95 {np.percentile(latencies, 95):.1f} ms. "
        f"Mean cost per query: ${np.mean([row['cost'] for row in rows]):.6f}."
    )
    if ablation:
        lines += [
            f"\n## Retrieval ablation\n\n| retriever | recall@{k} | MRR | p50 ms |",
            "|---|---:|---:|---:|",
        ]
        lines += [f"| {name} | {r:.3f} | {mrr:.3f} | {ms:.1f} |" for name, r, mrr, ms in ablation]
    return "\n".join(lines) + "\n"


def render_details(rows: list[dict]) -> str:
    lines = [
        "\n## Per question\n",
        "| id | recall | RR | faithfulness | relevance | similarity | ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['recall']:.2f} | {row['rr']:.2f} | {row['faithfulness']:.2f} "
            f"| {row['relevance']:.3f} | {row['similarity']:.3f} | {row['ms']:.1f} |"
        )
    misses = [row for row in rows if row["recall"] < 1 or row["faithfulness"] < 1]
    if misses:
        lines.append("\n## Questions with recall or faithfulness below 1\n")
    for row in misses:
        answer = " ".join(row["answer"].split())
        lines.append(
            f"- **{row['id']}** {row['question']} (recall {row['recall']:.2f}, "
            f"faithfulness {row['faithfulness']:.2f})  \n  Answer: {answer}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the pipeline against the golden set.")
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("evals/report.md"))
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    with RAGPipeline.from_settings() as pipeline:
        rows = evaluate(pipeline, golden, args.k, args.mode, args.rerank)
        ablation = run_ablation(pipeline, golden, args.k) if args.ablation else []
        generator = pipeline.generator.name

    scores = summarize(rows, args.k)
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    gate_table, failed = gate(scores, baseline)
    models = ", ".join(sorted({row["model"] for row in rows}))
    config = (
        f"Generator `{generator}` ({models}), k={args.k}, mode `{args.mode}`, "
        f"rerank {'on' if args.rerank else 'off'}, {len(rows)} questions, "
        f"faithfulness threshold {metrics.FAITHFULNESS_THRESHOLD}."
    )
    summary = render_summary(config, gate_table, rows, ablation, args.k)
    args.report.write_text(summary + render_details(rows), encoding="utf-8")
    print(summary)
    print(f"Report written to {args.report}")

    if args.update_baseline:
        tolerance = baseline.get("tolerance", DEFAULT_TOLERANCE) if baseline else DEFAULT_TOLERANCE
        content = json.dumps({"tolerance": tolerance, "metrics": scores}, indent=2)
        args.baseline.write_text(content + "\n", encoding="utf-8")
        print(f"Baseline written to {args.baseline}")
        return 0
    if baseline is None:
        print(f"No baseline at {args.baseline}, skipping the regression gate.")
        print("Create one with: python -m evals.run --update-baseline")
        return 0
    if failed:
        print("Quality gate failed: a metric fell below its baseline minus the tolerance.")
        return 1
    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
