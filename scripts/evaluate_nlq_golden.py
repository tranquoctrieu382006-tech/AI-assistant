"""Execute the NLQ golden set's reference SQL in a read-only transaction."""
import json
from pathlib import Path

import psycopg

from app.core.config import settings


def evaluate() -> dict:
    cases = json.loads(Path("data/nlq_golden_set.json").read_text(encoding="utf-8"))
    passed = 0
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            for case in cases:
                sql = case["sql"].strip()
                if not sql.upper().startswith("SELECT"):
                    raise ValueError(f"{case['id']} reference SQL must be read-only SELECT")
                cursor.execute(sql)
                columns = [column.name for column in cursor.description]
                actual = [dict(zip(columns, row)) for row in cursor.fetchall()]
                expected = [case["expected"]]
                ok = actual == expected
                passed += int(ok)
                print(f"{case['id']}: {'PASS' if ok else 'FAIL'} expected={expected} actual={actual}")
    total = len(cases)
    return {"cases": total, "passed": passed, "accuracy": passed / total if total else 0}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
