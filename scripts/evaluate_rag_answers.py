"""Evaluate live RAG API answers against lightweight golden answer strings."""
import json
from pathlib import Path

import httpx


def evaluate(base_url: str = "http://127.0.0.1:8000") -> dict:
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    passed = 0
    with httpx.Client(timeout=180) as client:
        for case in cases:
            response = client.post(f"{base_url}/api/rag/ask", json={"question": case["question"]})
            response.raise_for_status()
            answer = response.json()["answer"].casefold()
            expected = case["answer_contains"].casefold()
            ok = expected in answer
            passed += int(ok)
            print(f"{case['id']}: {'PASS' if ok else 'FAIL'}")
    return {"answer_cases": len(cases), "answer_contains_passed": passed,
            "answer_contains_rate": passed / len(cases) if cases else 0,
            "metric_note": "Substring smoke check only; not a semantic quality score."}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
