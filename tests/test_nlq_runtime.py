"""Runtime coverage for NLQ SQL safety, schema linking, and graph branches."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from fastapi import HTTPException

from app.api.nlq import ask_database
from app.core import nlq_graph
from app.core.config import settings
from app.core.nlq_graph import NlqRequestTimeout, _make_graph, validate_sql
from app.core.nlq_schema import select_relevant_tables
from app.schemas.nlq import NlqAskRequest


ALL_TABLES = {
    "categories", "suppliers", "products", "customers", "warehouses",
    "orders", "order_items", "inventory", "stock_movements",
}


def _schema():
    def col(name):
        return {"name": name, "type": "text", "tags": []}

    return [
        {"name": "categories", "columns": [col("category_id"), col("name")], "relations": ["products"]},
        {"name": "suppliers", "columns": [col("supplier_id"), col("city")], "relations": ["products"]},
        {"name": "products", "columns": [col("product_id"), col("category_id"), col("supplier_id")], "relations": ["categories", "suppliers", "order_items"]},
        {"name": "customers", "columns": [col("customer_id"), col("member_tier"), col("city")], "relations": ["orders"]},
        {"name": "warehouses", "columns": [col("warehouse_id"), col("city")], "relations": ["inventory", "stock_movements"]},
        {"name": "orders", "columns": [col("order_id"), col("customer_id"), col("order_date"), col("status")], "relations": ["customers", "order_items"]},
        {"name": "order_items", "columns": [col("order_id"), col("product_id"), col("quantity"), col("unit_price")], "relations": ["orders", "products"]},
        {"name": "inventory", "columns": [col("warehouse_id"), col("product_id"), col("on_hand")], "relations": ["warehouses", "products"]},
        {"name": "stock_movements", "columns": [col("warehouse_id"), col("product_id"), col("movement_type")], "relations": ["warehouses", "products"]},
    ]


def test_validate_all_current_golden_sql():
    cases = json.loads(Path("data/nlq_golden_set.json").read_text(encoding="utf-8"))
    for case in cases:
        validate_sql(case["sql"], ALL_TABLES, question=case["question"])


def test_revenue_guard_is_scoped_to_revenue_intent():
    revenue_sql = (
        "SELECT SUM(oi.quantity * oi.unit_price) AS revenue "
        "FROM order_items oi JOIN orders o USING (order_id)"
    )
    with pytest.raises(ValueError, match="loại orders.status"):
        validate_sql(revenue_sql, {"orders", "order_items"}, question="Doanh thu hàng hóa là bao nhiêu?")

    # Ordered value and average order value do not inherit the revenue policy.
    validate_sql(
        "SELECT SUM(oi.quantity * oi.unit_price) AS spend FROM order_items oi",
        {"order_items"},
        question="Tổng giá trị hàng đã đặt là bao nhiêu?",
    )
    validate_sql(
        "SELECT AVG(order_total) FROM (SELECT order_id, SUM(quantity * unit_price) order_total "
        "FROM order_items GROUP BY order_id) totals",
        {"order_items"},
        question="Giá trị trung bình đơn hàng là bao nhiêu?",
    )


def test_guard_accepts_cast_date_literal_and_to_char_golden_queries():
    cases = json.loads(Path("data/nlq_golden_set.json").read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}
    for case_id in ("nlq-017", "nlq-019", "nlq-030"):
        validate_sql(by_id[case_id]["sql"], ALL_TABLES, question=by_id[case_id]["question"])


def test_validate_sql_rejects_whole_row_alias_or_table_projection():
    # Tham chiếu cả dòng qua table alias
    with pytest.raises(ValueError, match="Không cho phép chọn trực tiếp cả dòng hoặc alias"):
        validate_sql("SELECT c FROM customers c", ALL_TABLES)

    # Tham chiếu cả dòng qua tên bảng
    with pytest.raises(ValueError, match="Không cho phép chọn trực tiếp cả dòng hoặc alias"):
        validate_sql("SELECT customers FROM customers", ALL_TABLES)


def test_validate_sql_rejects_cte_schema_injection():
    # CTE trùng tên bảng bị cấm, rồi tham chiếu bảng thật kèm schema
    cte_sql = "WITH rag_chunks AS (SELECT 1 AS a) SELECT a FROM orders CROSS JOIN public.rag_chunks"
    with pytest.raises(ValueError, match="Truy vấn tham chiếu bảng ngoài schema"):
        validate_sql(cte_sql, ALL_TABLES)

    # Schema không hợp lệ ngoài public
    foreign_schema_sql = "SELECT order_id FROM other_db.orders"
    with pytest.raises(ValueError, match="Truy vấn tham chiếu schema hoặc catalog không được phép"):
        validate_sql(foreign_schema_sql, ALL_TABLES)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Số đơn theo tháng trong năm 2026 là bao nhiêu?", {"orders", "customers", "order_items"}),
        ("Hạng thành viên nào phổ biến nhất?", {"customers", "orders"}),
        ("Thành phố nào có nhiều khách hàng nhất?", {"customers", "suppliers", "warehouses"}),
        ("Nhập kho?", {"stock_movements", "warehouses", "products"}),
        ("Đơn hàng theo trạng thái?", {"orders", "stock_movements"}),
    ],
)
def test_schema_fallback_resolves_vietnamese_phrases(question, expected):
    selected = select_relevant_tables(question, _schema(), llm=None)
    assert expected <= {table["name"] for table in selected}


def test_schema_fallback_does_not_guess_when_question_has_no_schema_terms():
    assert select_relevant_tables("Thời tiết ngày mai thế nào?", _schema(), llm=None) == []


class FakeNlqModel:
    def __init__(self, *, invalid_first_sql=False, out_of_scope=False):
        self.invalid_first_sql = invalid_first_sql
        self.out_of_scope = out_of_scope
        self.generated = False
        self.calls = []

    def invoke(self, prompt, config=None):
        text = prompt[0].content if isinstance(prompt, list) else str(prompt)
        self.calls.append(text)
        if "Choose the minimum table names" in text:
            content = "[]" if self.out_of_scope else '["orders"]'
        elif "Bạn là bộ sinh SQL PostgreSQL" in text:
            self.generated = True
            content = "DELETE FROM orders;" if self.invalid_first_sql else "SELECT COUNT(*) AS order_count FROM orders;"
        elif "Sửa câu SQL PostgreSQL" in text:
            content = "SELECT COUNT(*) AS order_count FROM orders;"
        elif "Trả lời ngắn gọn bằng tiếng Việt" in text:
            content = "Có 300 đơn hàng."
        else:
            raise AssertionError("Unexpected NLQ prompt stage")
        return SimpleNamespace(content=content)


def test_graph_runs_generate_validate_execute_interpret(monkeypatch):
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    monkeypatch.setattr(
        nlq_graph, "_run_readonly",
        lambda sql: (["order_count"], [{"order_count": 300}], False),
    )
    model = FakeNlqModel()
    result = _make_graph(model).invoke({
        "question": "Có bao nhiêu đơn hàng?", "repair_count": 0,
        "repaired": False, "callbacks": [],
    })
    assert result["rows"] == [{"order_count": 300}]
    assert result["answer"] == "Có 300 đơn hàng."
    assert result["repaired"] is False
    assert len(model.calls) == 3


def test_graph_terminates_immediately_when_database_unavailable(monkeypatch):
    import psycopg
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    def fail_readonly(sql):
        raise psycopg.OperationalError("connection to server at 127.0.0.1 failed")

    monkeypatch.setattr(nlq_graph, "_run_readonly", fail_readonly)
    model = FakeNlqModel()
    result = _make_graph(model).invoke({
        "question": "Có bao nhiêu đơn hàng?", "repair_count": 0,
        "repaired": False, "callbacks": [],
    })
    # Khẳng định dừng ngay khi DB lỗi mạng, không gọi LLM để repair_sql
    assert "OperationalError" in result.get("error", "")
    assert not any("Sửa câu SQL" in call for call in model.calls)


def test_graph_repairs_invalid_sql_once(monkeypatch):
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    monkeypatch.setattr(
        nlq_graph, "_run_readonly",
        lambda sql: (["order_count"], [{"order_count": 300}], False),
    )
    model = FakeNlqModel(invalid_first_sql=True)
    result = _make_graph(model).invoke({
        "question": "Có bao nhiêu đơn hàng?", "repair_count": 0,
        "repaired": False, "callbacks": [],
    })
    assert result["repaired"] is True
    assert result["repair_count"] == 1
    assert result["rows"] == [{"order_count": 300}]


def test_graph_returns_out_of_scope_without_generating_sql(monkeypatch):
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    model = FakeNlqModel(out_of_scope=True)
    result = _make_graph(model).invoke({
        "question": "Thời tiết ngày mai thế nào?", "repair_count": 0,
        "repaired": False, "callbacks": [],
    })
    assert result["sql"] == ""
    assert result["rows"] == []
    assert not model.generated


def test_graph_honors_expired_request_deadline(monkeypatch):
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    model = FakeNlqModel()
    with pytest.raises(NlqRequestTimeout):
        _make_graph(model).invoke({
            "question": "Có bao nhiêu đơn hàng?", "repair_count": 0,
            "repaired": False, "callbacks": [],
            "deadline_monotonic": time.monotonic() - 1,
        })
    assert model.calls == []


def test_nlq_http_maps_request_deadline_to_504(monkeypatch):
    monkeypatch.setattr(settings, "nlq_database_url", "postgresql://readonly/test")
    def timed_out(*args, **kwargs):
        raise NlqRequestTimeout("deadline")
    monkeypatch.setattr("app.api.nlq.ask_nlq", timed_out)
    with pytest.raises(HTTPException) as exc:
        ask_database(NlqAskRequest(question="Có bao nhiêu đơn hàng?"))
    assert exc.value.status_code == 504


def test_nlq_http_returns_503_when_database_is_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "nlq_database_url", "")
    with pytest.raises(HTTPException) as exc:
        ask_database(NlqAskRequest(question="Có bao nhiêu đơn hàng?"))
    assert exc.value.status_code == 503


def test_nlq_http_returns_graph_response(monkeypatch):
    monkeypatch.setattr(settings, "nlq_database_url", "postgresql://readonly/test")
    monkeypatch.setattr("app.api.nlq.ask_nlq", lambda *args, **kwargs: {
        "sql": "SELECT COUNT(*) FROM orders",
        "columns": ["count"],
        "rows": [{"count": 300}],
        "answer": "Có 300 đơn hàng.",
        "repaired": False,
        "truncated": False,
    })
    response = ask_database(NlqAskRequest(question="Có bao nhiêu đơn hàng?"))
    assert response.rows == [{"count": 300}]
    assert response.truncated is False


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "SELECT pg_sleep(60) FROM orders",
        "SELECT * FROM rag_chunks",
    ],
)
def test_sql_guard_blocks_injection_payloads(sql):
    with pytest.raises(ValueError):
        validate_sql(sql, ALL_TABLES, question="Có bao nhiêu đơn hàng?")


def test_validate_sql_blocks_aliased_sensitive_columns():
    # Attempting to alias a sensitive column like phone or email should be blocked
    aliased_sql = "SELECT phone AS contact_num FROM customers"
    with pytest.raises(ValueError, match="nhạy cảm"):
        validate_sql(aliased_sql, ALL_TABLES)

    aliased_cte = "WITH safe_names AS (SELECT email AS contact_mail FROM customers) SELECT contact_mail FROM safe_names"
    with pytest.raises(ValueError, match="nhạy cảm"):
        validate_sql(aliased_cte, ALL_TABLES)


def test_validate_sql_accepts_status_whitelist_for_revenue():
    # Revenue query with active status whitelist should be accepted
    sql = (
        "SELECT SUM(oi.quantity * oi.unit_price) AS revenue "
        "FROM orders o JOIN order_items oi ON o.order_id = oi.order_id "
        "WHERE o.status IN ('mới', 'đã xác nhận', 'đang giao', 'đã giao')"
    )
    validated = validate_sql(sql, ALL_TABLES, question="Doanh thu hàng hóa tháng này?")
    assert "revenue" in validated
