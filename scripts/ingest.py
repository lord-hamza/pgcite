import argparse
import sys
from pathlib import Path

from pgcite.chunking import load_markdown
from pgcite.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a folder of markdown files.")
    parser.add_argument("path", nargs="?", type=Path, default=Path("data/docs"))
    args = parser.parse_args()

    documents = load_markdown(args.path)
    if not documents:
        sys.exit(f"No markdown files found in {args.path}")
    with RAGPipeline.from_settings() as pipeline:
        chunks = pipeline.ingest(documents)
    print(f"Indexed {len(documents)} documents into {chunks} chunks")


if __name__ == "__main__":
    main()
