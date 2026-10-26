import argparse
import json
import time
from pathlib import Path

import numpy as np

from pgcite.pipeline import RAGPipeline

STAGES = ["embed", "bm25", "vector", "fuse", "rerank", "generate", "total"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure per-stage latency and cost of ask().")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    lines = args.golden.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line)["question"] for line in lines if line.strip()]

    with RAGPipeline.from_settings() as pipeline:
        pipeline.ask(questions[0], k=args.k)
        started = time.perf_counter()
        results = [pipeline.ask(q, k=args.k) for _ in range(args.runs) for q in questions]
        elapsed = time.perf_counter() - started
        generator = pipeline.generator.name

    print("| stage | p50 ms | p95 ms | mean ms |")
    print("|---|---:|---:|---:|")
    for stage in STAGES:
        values = [result.timings_ms[stage] for result in results if stage in result.timings_ms]
        if values:
            p50, p95 = np.percentile(values, [50, 95])
            print(f"| {stage} | {p50:.1f} | {p95:.1f} | {np.mean(values):.1f} |")

    answers = [result.answer for result in results]
    models = ", ".join(sorted({answer.model for answer in answers}))
    print(
        f"\n{len(results)} queries, generator {generator} ({models}), "
        f"mean tokens {np.mean([a.input_tokens for a in answers]):.0f} in / "
        f"{np.mean([a.output_tokens for a in answers]):.0f} out, "
        f"mean cost ${np.mean([a.cost_usd for a in answers]):.6f}/query, "
        f"{len(results) / elapsed:.1f} queries/s sequential"
    )


if __name__ == "__main__":
    main()
