"""Check SQL guards and bounded auto-repair against the sample read-only database."""
from types import SimpleNamespace

import psycopg

from app.core.config import settings
from app.core.nlq_graph import _make_graph, validate_sql
from app.core.nlq_schema import load_schema


class FakeModel:
    def invoke(self, prompt, config=None):
        text = prompt[0].content if isinstance(prompt, list) else str(prompt)
        if "Choose the minimum table names" in text:
            response = '["orders"]'
        elif "Bạn là bộ sinh SQL PostgreSQL" in text:
            response = "DELETE FROM orders;"
        elif "Sửa câu SQL PostgreSQL" in text:
            response = "SELECT COUNT(*) FROM orders;"
        elif "Trả lời ngắn gọn bằng tiếng Việt" in text:
            response = "Có 300 đơn hàng."
        else:
            raise AssertionError("Unexpected prompt stage.")
        return SimpleNamespace(content=response)


def main() -> None:
    allowed = {"orders", "customers"}
    assert "LIMIT 200" in validate_sql("SELECT COUNT(*) FROM orders", allowed)
    for unsafe_sql in (
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "SELECT pg_sleep(60) FROM orders",
        "SELECT * FROM rag_chunks",
    ):
        try:
            validate_sql(unsafe_sql, allowed)
        except ValueError:
            continue
        raise AssertionError(f"Unsafe SQL unexpectedly accepted: {unsafe_sql}")
    print("SQL validator: safe SELECT accepted; DELETE/multiple/system/out-of-schema rejected.")

    if not settings.nlq_database_url.strip():
        raise RuntimeError("NLQ_DATABASE_URL is required for this read-only check.")
    with psycopg.connect(settings.nlq_database_url, connect_timeout=3) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        has_email = conn.execute(
            "SELECT has_column_privilege(current_user, 'public.customers', 'email', 'SELECT')"
        ).fetchone()[0]
        has_full_name = conn.execute(
            "SELECT has_column_privilege(current_user, 'public.customers', 'full_name', 'SELECT')"
        ).fetchone()[0]
    assert not has_email and not has_full_name
    print("Database grants: customer email/full_name SELECT privileges are denied.")

    settings.nlq_max_repairs = 1
    graph = _make_graph(FakeModel())
    result = graph.invoke({
        "question": "Có bao nhiêu đơn hàng?",
        "repair_count": 0,
        "repaired": False,
        "callbacks": [],
    })
    assert result.get("repaired") is True
    assert result["repair_count"] == 1
    assert result["rows"] == [{"count": 300}]
    print("Auto-repair: blocked DELETE, repaired to SELECT, returned one read-only result.")


if __name__ == "__main__":
    main()
