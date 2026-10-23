"""Evaluate model-generated SQL against the NLQ golden set, read-only."""
import argparse
import json
from pathlib import Path
import json

import psycopg

from app.core.config import settings
from app.core.llm import get_llm
from scripts.evaluate_nlq_golden import is_read_only_query, results_match


def _extract_sql(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    if text[:4].casefold() == "sql\n":
        text = text[4:].strip()
    return text


def _execution_results_match(actual: list[dict], expected: list[dict]) -> bool:
    """Ignore output aliases and row ordering while comparing SQL result values."""
    if results_match(actual, expected):
        return True
    actual_rows = [{str(i): value for i, value in enumerate(row.values())} for row in actual]
    expected_rows = [{str(i): value for i, value in enumerate(row.values())} for row in expected]
    key = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return results_match(sorted(actual_rows, key=key), sorted(expected_rows, key=key))


def evaluate(model_role: str = "local", limit: int | None = None) -> dict:
    cases = json.loads(Path("data/nlq_golden_set.json").read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]
    schema = Path("data/demo_sales_warehouse/data_dictionary.md").read_text(encoding="utf-8-sig")
    llm = get_llm(model_role, temperature=0)
    passed = 0
    for case in cases:
        prompt = (
            "Generate one PostgreSQL SELECT query for the user's question. "
            "Return only SQL, no explanation. Use only tables and columns in the schema. "
            "Do not modify data.\n\nSCHEMA:\n" + schema +
            "\nQUESTION:\n" + case["question"]
        )
        try:
            sql = _extract_sql(str(llm.invoke(prompt).content))
            # PostgreSQL read-only mode is a second guard; reject multi-statement output too.
            if not is_read_only_query(sql) or sql.rstrip().rstrip(";").find(";") >= 0:
                print(f"{case['id']}: REJECTED non-single-read-only SQL")
                continue
            with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
                conn.execute("SET TRANSACTION READ ONLY")
                conn.execute("SET LOCAL statement_timeout = '10s'")
                cursor = conn.execute(sql)
                actual = [dict(zip((col.name for col in cursor.description), row))
                          for row in cursor.fetchall()]
            expected_value = case["expected"]
            expected = expected_value if isinstance(expected_value, list) else [expected_value]
            ok = _execution_results_match(actual, expected)
            passed += int(ok)
            print(f"{case['id']}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(f"  generated_sql: {sql}")
                print(f"  actual_rows: {actual}")
        except Exception as exc:
            print(f"{case['id']}: ERROR {type(exc).__name__}")
    return {"model_role": model_role, "cases": len(cases), "execution_correct": passed,
            "execution_accuracy": passed / len(cases) if cases else 0,
            "model_evaluated": True, "trace_uploaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-role", choices=("local", "fast", "strong", "default"), default="local")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_role, args.limit), ensure_ascii=False, indent=2))
