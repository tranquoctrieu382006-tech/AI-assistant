"""Report retrieval Hit@k and MRR against data/rag_golden_set.json."""
import json
from pathlib import Path, PurePosixPath

from app.core.config import settings


def source_matches(actual: str, expected: str) -> bool:
    """Compare corpus-relative source paths, not potentially ambiguous basenames."""
    actual_path = PurePosixPath(actual.replace("\\", "/"))
    expected_path = PurePosixPath(expected.replace("\\", "/"))
    return actual_path == expected_path


def evaluate(k: int = 5) -> dict:
    from app.core.vector_store import similarity_search
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    hits, reciprocal_ranks = 0, []
    answerable = [case for case in cases if case["source"] is not None]
    abstentions = [case for case in cases if case["source"] is None]
    abstention_correct = 0
    for case in answerable:
        results = similarity_search(case["question"], k=k)
        expected = case["source"]
        rank = next((i for i, item in enumerate(results, 1)
                     if expected and source_matches(item["source"], expected)), None)
        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1 / rank if rank else 0)
        print(f"{case['id']}: {'HIT' if hit else 'MISS'} rank={rank} top={results[0]['source'] if results else '-'}")
    for case in abstentions:
        results = similarity_search(case["question"], k=k)
        abstained = not results or results[0]["score"] < settings.rag_min_relevance_score
        abstention_correct += int(abstained)
        print(f"{case['id']}: {'ABSTAIN PASS' if abstained else 'ABSTAIN FAIL'} "
              f"top_score={results[0]['score'] if results else '-'}")
    total = len(answerable)
    return {"answerable_cases": total, f"hit@{k}": hits / total if total else 0,
            "mrr": sum(reciprocal_ranks) / total if total else 0,
            "abstention_cases": len(abstentions),
            "abstention_accuracy": abstention_correct / len(abstentions) if abstentions else None}


if __name__ == "__main__":
    print(json.dumps(evaluate(settings.rag_top_k), ensure_ascii=False, indent=2))
