"""Execute the NLQ golden set's reference SQL in a read-only transaction."""
import json
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import psycopg

from app.core.config import settings


_READ_ONLY_SQL_START = re.compile(
    r"\A(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*(SELECT|WITH)\b",
    re.IGNORECASE | re.DOTALL,
)


def is_read_only_query(sql: str) -> bool:
    """Allow SELECT and CTE queries; PostgreSQL read-only transaction is the backstop."""
    return _READ_ONLY_SQL_START.match(sql) is not None


def _normalize_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def results_match(actual: list[dict], expected: list[dict]) -> bool:
    return _normalize_value(actual) == _normalize_value(expected)


def evaluate() -> dict:
    cases = json.loads(Path("data/nlq_golden_set.json").read_text(encoding="utf-8"))
    passed = 0
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            for case in cases:
                sql = case["sql"].strip()
                if not is_read_only_query(sql):
                    raise ValueError(f"{case['id']} reference SQL must start with SELECT or WITH")
                cursor.execute(sql)
                columns = [column.name for column in cursor.description]
                actual = [dict(zip(columns, row)) for row in cursor.fetchall()]
                expected = [case["expected"]]
                ok = results_match(actual, expected)
                passed += int(ok)
                print(f"{case['id']}: {'PASS' if ok else 'FAIL'} expected={expected} actual={actual}")
    total = len(cases)
    return {"cases": total, "passed": passed, "accuracy": passed / total if total else 0}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
