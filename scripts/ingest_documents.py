"""Run once to ingest the configured document directory into pgvector."""
import argparse
import json

from app.core.ingestion import ingest_directory
from app.core.vector_store import close_pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("fixed_size", "heading"),
        default="heading",
        help="Choose fixed-size overlap chunks or section-bounded chunks.",
    )
    args = parser.parse_args()
    try:
        print(json.dumps(
            ingest_directory(strategy=args.strategy),
            ensure_ascii=False,
            indent=2,
        ))
    finally:
        close_pool()
