"""Evaluate model-generated SQL against the NLQ golden set, read-only."""
import argparse
import json
from pathlib import Path
import re

import psycopg

from app.core.config import settings
from app.core.llm import get_llm
from scripts.evaluate_nlq_golden import is_read_only_query, results_match


def _sql_features(sql: str) -> tuple[int, bool, bool]:
    """Return external semicolon count, trailing code, and top-level ORDER BY."""
    semicolons = 0
    trailing_code = False
    tokens: list[tuple[str, int]] = []
    depth = 0
    index = 0
    after_semicolon = False

    while index < len(sql):
        char = sql[index]
        if char.isspace():
            index += 1
            continue
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline < 0 else newline + 1
            continue
        if sql.startswith("/*", index):
            comment_depth = 1
            index += 2
            while index < len(sql) and comment_depth:
                if sql.startswith("/*", index):
                    comment_depth += 1
                    index += 2
                elif sql.startswith("*/", index):
                    comment_depth -= 1
                    index += 2
                else:
                    index += 1
            continue
        if after_semicolon:
            trailing_code = True
        if char == "'":
            escape_string = (
                index > 0
                and sql[index - 1].casefold() == "e"
                and (index == 1 or not (sql[index - 2].isalnum() or sql[index - 2] == "_"))
            )
            index += 1
            while index < len(sql):
                if sql[index] == "'":
                    if index + 1 < len(sql) and sql[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                if escape_string and sql[index] == "\\" and index + 1 < len(sql):
                    index += 2
                else:
                    index += 1
            continue
        if char == '"':
            index += 1
            while index < len(sql):
                if sql[index] == '"':
                    if index + 1 < len(sql) and sql[index + 1] == '"':
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue
        if char == "$":
            delimiter = re.match(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$", sql[index:])
            if delimiter:
                marker = delimiter.group()
                end = sql.find(marker, index + len(marker))
                index = len(sql) if end < 0 else end + len(marker)
                continue
        if char == ";":
            semicolons += 1
            after_semicolon = True
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(sql) and (sql[index].isalnum() or sql[index] in "_$"):
                index += 1
            tokens.append((sql[start:index].casefold(), depth))
            continue
        index += 1

    top_level_words = [word for word, token_depth in tokens if token_depth == 0]
    has_order_by = any(
        first == "order" and second == "by"
        for first, second in zip(top_level_words, top_level_words[1:])
    )
    return semicolons, trailing_code, has_order_by


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
            semicolons, trailing_code, ordered = _sql_features(sql)
            # The database read-only transaction is a second guard.
            if (not is_read_only_query(sql)
                    or semicolons > 1
                    or trailing_code):
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
