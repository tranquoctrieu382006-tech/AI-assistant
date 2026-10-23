"""Evaluate model-generated SQL against the NLQ golden set, read-only."""
import argparse
import json
from pathlib import Path
import re

import psycopg
from sqlglot import exp, parse

from app.core.config import settings
from app.core.llm import get_llm
from scripts.evaluate_nlq_golden import is_read_only_query, results_match


def _sql_features(sql: str) -> tuple[int, bool]:
    """Return parsed statement count and whether the outer query orders rows."""
    statements = [
        statement for statement in parse(sql, read="postgres")
        if statement is not None and not isinstance(statement, exp.Semicolon)
    ]
    has_order_by = len(statements) == 1 and statements[0].args.get("order") is not None
    return len(statements), has_order_by


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


def _execution_results_match(
    actual: list[dict], expected: list[dict], *, ordered: bool = False
) -> bool:
    """Ignore aliases, and ignore row order only when SQL does not specify it."""
    actual_rows = [{str(i): value for i, value in enumerate(row.values())} for row in actual]
    expected_rows = [{str(i): value for i, value in enumerate(row.values())} for row in expected]
    if ordered:
        return results_match(actual_rows, expected_rows)
    key = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return results_match(sorted(actual_rows, key=key), sorted(expected_rows, key=key))


def _load_schema_ddl() -> str:
    """Use the seed's CREATE TABLE statements as the SQL-generation schema."""
    seed = Path("data/demo_sales_warehouse/schema_seed.sql").read_text(encoding="utf-8-sig")
    statements = re.findall(r"\bCREATE\s+TABLE\b.*?;", seed, flags=re.IGNORECASE | re.DOTALL)
    if not statements:
        raise ValueError("No CREATE TABLE statements found in data/demo_sales_warehouse/schema_seed.sql")
    return "\n".join(statements)


def evaluate(model_role: str = "local", limit: int | None = None) -> dict:
    cases = json.loads(Path("data/nlq_golden_set.json").read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]
    schema = _load_schema_ddl()
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
            statement_count, ordered = _sql_features(sql)
            # The database read-only transaction is a second guard.
            if (not is_read_only_query(sql)
                    or statement_count != 1):
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
            ok = _execution_results_match(actual, expected, ordered=ordered)
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
