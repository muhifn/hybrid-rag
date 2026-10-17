"""Build the vector store and the graph backend from data/.

    python -m scripts.ingest              # rebuild everything (default)
    python -m scripts.ingest --keep       # upsert into the existing collection

Embedded Qdrant locks its folder to one process. If the API is running in
embedded mode, use `POST /admin/ingest` instead of this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from app.config import get_settings
from app.indexing import run_ingestion
from app.llm import build_embeddings, embedding_dimension
from app.vectorstore import VectorIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="do not clear the collection first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    settings = get_settings()
    embeddings = build_embeddings(settings)

    try:
        index = VectorIndex(settings, embeddings, embedding_dimension(embeddings))
    except RuntimeError as exc:
        if "already accessed" in str(exc):
            print("\nQdrant storage is locked by another process (is the API running?).\n"
                  "Use `POST /admin/ingest` on the running API, or stop it first.", file=sys.stderr)
            return 1
        raise

    try:
        stats = run_ingestion(settings, index, recreate=not args.keep)
    finally:
        index.close()
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
