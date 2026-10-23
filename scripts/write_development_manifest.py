"""Write the frozen-corpus/development (M1) fingerprint for RAG evaluation."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data/demo_sales_warehouse"
MANIFEST = ROOT / "eval/development_manifest_m1.json"
EXCLUDED = {"README.md", "data_dictionary.md", "nguon_dataset.md", "schema_seed.sql"}
EVALUATION_FILES = [
    "scripts/evaluate_rag_retrieval.py",
    "scripts/rag_eval_metrics.py",
    "scripts/validate_golden_data.py",
    "scripts/write_development_manifest.py",
    "tests/test_golden_data_validation.py",
    "docs/rag_eval_protocol.md",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    corpus = sorted(
        path for path in CORPUS_DIR.iterdir()
        if path.is_file() and path.name not in EXCLUDED
    )
    rag_path = ROOT / "data/rag_golden_set.json"
    nlq_path = ROOT / "data/nlq_golden_set.json"
    rag = json.loads(rag_path.read_text(encoding="utf-8"))
    config = {
        "document_dir": "data/demo_sales_warehouse",
        "excluded_files": sorted(EXCLUDED),
        "chunking_strategy": "heading",
        "chunk_size_chars": 900,
        "chunk_overlap_chars": 150,
        "top_k": 5,
        "embedding_model": "bge-m3",
        "primary_metric": "context_recall@5",
        "risk_metric": "false_positive_count_by_abstention_type",
    }
    manifest = {
        "manifest": "M1",
        "purpose": "Frozen corpus, development data, scorer, and retrieval configuration before held-out creation.",
        "configuration": config,
        "corpus_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in corpus
        },
        "development_sets": {
            "data/rag_golden_set.json": {
                "sha256": sha256(rag_path),
                "case_count": len(rag),
                "categories": dict(sorted(Counter(case["category"] for case in rag).items())),
            },
            "data/nlq_golden_set.json": {
                "sha256": sha256(nlq_path),
                "case_count": len(json.loads(nlq_path.read_text(encoding="utf-8"))),
            },
        },
        "evaluation_code": {
            rel: sha256(ROOT / rel) for rel in EVALUATION_FILES
        },
        "held_out": None,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT).as_posix()}")
    print(f"Corpus files: {len(corpus)}; RAG development cases: {len(rag)}")


if __name__ == "__main__":
    main()
