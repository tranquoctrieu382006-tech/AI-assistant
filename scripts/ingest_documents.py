"""Run once to ingest the configured document directory into pgvector."""
import json

from app.core.ingestion import ingest_directory
from app.core.vector_store import close_pool


if __name__ == "__main__":
    try:
        print(json.dumps(ingest_directory(), ensure_ascii=False, indent=2))
    finally:
        close_pool()
