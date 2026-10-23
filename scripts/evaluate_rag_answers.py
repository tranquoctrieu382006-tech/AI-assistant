"""Evaluate live RAG API answers against lightweight golden answer strings."""
import json
from pathlib import Path

import httpx


def evaluate(base_url: str = "http://127.0.0.1:8000") -> dict:
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    passed = 0
    category_results: dict[str, dict[str, int]] = {}
    with httpx.Client(timeout=180) as client:
        for case in cases:
            response = client.post(f"{base_url}/api/rag/ask", json={"question": case["question"]})
            response.raise_for_status()
            answer = response.json()["answer"].casefold()
            expected = case["answer_contains"]
            expected_parts = expected if isinstance(expected, list) else [expected]
            ok = all(part.casefold() in answer for part in expected_parts)
            passed += int(ok)
            category = case.get("category", "simple")
            counts = category_results.setdefault(category, {"cases": 0, "passed": 0})
            counts["cases"] += 1
            counts["passed"] += int(ok)
            print(f"{case['id']}: {'PASS' if ok else 'FAIL'}")
    return {"answer_cases": len(cases), "answer_contains_passed": passed,
            "answer_contains_rate": passed / len(cases) if cases else 0,
            "by_category": {
                category: {
                    **counts,
                    "answer_contains_rate": counts["passed"] / counts["cases"],
                }
                for category, counts in sorted(category_results.items())
            },
            "metric_note": "Substring smoke check only; not a semantic quality score."}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
