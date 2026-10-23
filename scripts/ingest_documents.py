"""Run once to ingest the configured document directory into pgvector."""
import json

from app.core.ingestion import ingest_directory


if __name__ == "__main__":
    print(json.dumps(ingest_directory(), ensure_ascii=False, indent=2))
