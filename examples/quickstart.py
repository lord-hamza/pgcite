import sys
from pathlib import Path

from pgcite.chunking import load_markdown
from pgcite.pipeline import RAGPipeline

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"
DEFAULT_QUESTION = (
    "How far back can we look in the webhook delivery logs compared with the API request log?"
)


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    with RAGPipeline.from_settings() as pipeline:
        chunks = pipeline.ingest(load_markdown(DOCS))
        result = pipeline.ask(question)

    answer = result.answer
    print(f"Indexed {chunks} chunks\n\nQ: {question}\nA: {answer.text}\n")
    for citation in answer.citations:
        print(f"  [{citation.ref}] {citation.title} ({citation.doc_id})")
    timings = ", ".join(f"{stage} {ms:.1f} ms" for stage, ms in result.timings_ms.items())
    print(f"\n{timings}\nmodel {answer.model}, cost ${answer.cost_usd:.6f}")


if __name__ == "__main__":
    main()
