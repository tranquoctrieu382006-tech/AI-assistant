"""Evaluate live RAG API answers against lightweight golden answer strings."""
import json
from pathlib import Path
import random
import time

import httpx

from app.core.config import settings


def _post_with_rate_limit_retry(
    client: httpx.Client,
    url: str,
    *,
    json_body: dict,
    headers: dict[str, str],
    max_retries: int = 5,
    sleep=time.sleep,
) -> httpx.Response:
    """Retry transient 429/5xx responses, honoring Retry-After when provided."""
    for attempt in range(max_retries + 1):
        response = client.post(url, json=json_body, headers=headers)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt == max_retries:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        try:
            delay = max(0.0, float(retry_after)) if retry_after else min(30.0, 2**attempt)
        except ValueError:
            delay = min(30.0, 2**attempt)
        sleep(delay + random.uniform(0, min(0.25, delay * 0.1)))
    raise RuntimeError("unreachable")


def evaluate(base_url: str = "http://127.0.0.1:8000") -> dict:
    if len(settings.app_api_key) < 32:
        raise RuntimeError("Set APP_API_KEY in .env before evaluating the protected RAG API.")
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    passed = 0
    category_results: dict[str, dict[str, int]] = {}
    with httpx.Client(timeout=180) as client:
        for index, case in enumerate(cases):
            # Stay below the app's default 60 requests/minute limit. Backoff
            # still handles shared-IP traffic and external provider throttling.
            if index:
                time.sleep(1.05)
            response = _post_with_rate_limit_retry(
                client,
                f"{base_url}/api/rag/ask",
                json_body={"question": case["question"]},
                headers={"X-API-Key": settings.app_api_key},
            )
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
            "metric_note": "Substring smoke check only; not a semantic quality score.",
            "rate_limit_note": "Requests paced at 1.05 seconds with bounded retry for 429/5xx."}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
