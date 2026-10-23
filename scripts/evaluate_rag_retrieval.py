"""Report retrieval Hit@k and MRR against data/rag_golden_set.json."""
import json
from pathlib import Path

from app.core.config import settings


def evaluate(k: int = 5) -> dict:
    from app.core.vector_store import similarity_search
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    hits, reciprocal_ranks = 0, []
    for case in cases:
        results = similarity_search(case["question"], k=k)
        expected = case["source"]
        rank = next((i for i, item in enumerate(results, 1)
                     if expected and Path(item["source"]).name == expected), None)
        if expected is None:
            rank = 1 if not results or results[0]["score"] < 0.25 else None
        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1 / rank if rank else 0)
        print(f"{case['id']}: {'HIT' if hit else 'MISS'} rank={rank} top={results[0]['source'] if results else '-'}")
    total = len(cases)
    return {"cases": total, f"hit@{k}": hits / total if total else 0,
            "mrr": sum(reciprocal_ranks) / total if total else 0}


if __name__ == "__main__":
    print(json.dumps(evaluate(settings.rag_top_k), ensure_ascii=False, indent=2))
