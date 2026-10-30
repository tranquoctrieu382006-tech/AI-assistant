"""Runtime coverage for NLQ SQL safety, schema linking, and graph branches."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import psycopg
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


def test_nlq_http_returns_503_on_database_operational_error(monkeypatch):
    monkeypatch.setattr(settings, "nlq_database_url", "postgresql://readonly/test")
    def db_error(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")
    monkeypatch.setattr("app.api.nlq.ask_nlq", db_error)
    with pytest.raises(HTTPException) as exc:
        ask_database(NlqAskRequest(question="Có bao nhiêu đơn hàng?"))
    assert exc.value.status_code == 503
    assert "unavailable" in exc.value.detail.lower()


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


def test_nlq_graph_with_langchain_fake_list_chat_model(monkeypatch):
    """Kiểm tra đồ thị NLQ thực thi chuẩn xác với FakeListChatModel chính thức của LangChain (TASK-AI23)."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    monkeypatch.setattr(
        nlq_graph, "_run_readonly",
        lambda sql: (["product_count"], [{"product_count": 150}], False),
    )
    # FakeListChatModel cung cấp chuỗi responses cho các node: link_schema -> generate_sql -> interpret
    fake_llm = FakeListChatModel(responses=[
        "orders, products",
        "```sql\nSELECT COUNT(*) FROM products;\n```",
        "Hiện tại hệ thống có 150 sản phẩm.",
        "orders, products",
        "```sql\nSELECT COUNT(*) FROM products;\n```",
        "Hiện tại hệ thống có 150 sản phẩm.",
    ])

    result = _make_graph(fake_llm).invoke({
        "question": "Có bao nhiêu sản phẩm?",
        "repair_count": 0,
        "repaired": False,
        "callbacks": [],
    })
    assert result["rows"] == [{"product_count": 150}]
    assert result["answer"] == "Hiện tại hệ thống có 150 sản phẩm."
    assert "products" in result["sql"]
    assert result["repaired"] is False

    # Kiểm tra trực tiếp hàm điểm vào ask_nlq (đảm bảo đọc đúng settings và không lỗi AttributeError)
    monkeypatch.setattr(nlq_graph, "get_llm", lambda *a, **kw: fake_llm)
    ask_result = nlq_graph.ask_nlq("Có bao nhiêu sản phẩm?")
    assert ask_result["rows"] == [{"product_count": 150}]
    assert ask_result["answer"] == "Hiện tại hệ thống có 150 sản phẩm."


@pytest.mark.parametrize(
    "malicious_sql,expected_error_pattern",
    [
        ("DELETE FROM orders", "Chỉ cho phép truy vấn SELECT hoặc WITH ... SELECT"),
        ("DROP TABLE orders", "Chỉ cho phép truy vấn SELECT hoặc WITH ... SELECT"),
        ("TRUNCATE TABLE orders", "Chỉ cho phép truy vấn SELECT hoặc WITH ... SELECT"),
        ("UPDATE customers SET total_spend = 0", "Chỉ cho phép truy vấn SELECT hoặc WITH ... SELECT"),
        ("INSERT INTO customers (name) VALUES ('Hacker')", "Chỉ cho phép truy vấn SELECT hoặc WITH ... SELECT"),
        ("WITH deleted AS (DELETE FROM orders RETURNING *) SELECT * FROM deleted", "Truy vấn chứa thao tác ghi hoặc lệnh không được phép"),
        ("SELECT * INTO new_orders FROM orders", "Không cho phép SELECT INTO hoặc khóa dòng"),
        ("SELECT pg_sleep(5)", "Truy vấn gọi hàm ngoài danh sách cho phép"),
        ("COPY orders TO '/tmp/leak.csv'", "Chỉ cho phép truy vấn SELECT hoặc WITH ... SELECT"),
        ("SELECT * FROM rag_chunks", "Truy vấn tham chiếu bảng ngoài schema được chọn"),
        ("SELECT phone, email FROM customers", "Truy vấn chọn cột nhạy cảm 'phone' và đã bị chặn"),
        ("SELECT 1; DELETE FROM orders", "Chỉ cho phép một câu lệnh SQL"),
    ],
)
def test_validate_sql_security_guard_blocks_malicious_constructs(malicious_sql, expected_error_pattern):
    allowed_tables = {"orders", "customers", "products", "order_items"}
    allowed_cols = {
        "orders": {"order_id", "customer_id", "total_amount", "status", "created_at"},
        "customers": {"customer_id", "name", "customer_segment", "city", "total_spend", "created_at"},
    }
    with pytest.raises(ValueError, match=expected_error_pattern):
        validate_sql(malicious_sql, allowed_tables, allowed_cols)


def test_nlq_sensitive_projection_is_refused_without_calling_the_model(monkeypatch):
    """Kiểm tra nhạy cảm phải đứng trước LLM chọn bảng, không tốn một lượt gọi."""
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)

    class ExplodingLLM:
        def invoke(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("LLM must not be called for a sensitive projection")

    res = _make_graph(ExplodingLLM()).invoke({
        "question": "Cho tôi số điện thoại của khách hàng",
        "repair_count": 0,
        "repaired": False,
        "callbacks": [],
    })
    assert res.get("status") == "restricted"
    assert res["sql"] == ""


def test_nlq_fails_fast_when_remaining_budget_cannot_fit_a_model_call(monkeypatch):
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    model = FakeNlqModel()
    with pytest.raises(NlqRequestTimeout):
        _make_graph(model).invoke({
            "question": "Doanh thu tháng này là bao nhiêu?",
            "repair_count": 0,
            "repaired": False,
            "callbacks": [],
            # Ít hơn ngưỡng 10 giây tối thiểu của Google GenAI.
            "deadline_monotonic": time.monotonic() + 5,
        })
    assert model.calls == []


def test_nlq_call_timeout_uses_remaining_budget_and_disables_retries(monkeypatch):
    monkeypatch.setattr(nlq_graph, "load_schema", _schema)
    monkeypatch.setattr(
        nlq_graph, "_run_readonly",
        lambda sql: (["order_count"], [{"order_count": 1}], False),
    )

    class RecordingModel(FakeNlqModel):
        def __init__(self):
            super().__init__()
            self.invoke_kwargs = []

        def invoke(self, prompt, config=None, **kwargs):
            self.invoke_kwargs.append(kwargs)
            return super().invoke(prompt, config=config)

    model = RecordingModel()
    _make_graph(model).invoke({
        "question": "Có bao nhiêu đơn hàng?",
        "repair_count": 0,
        "repaired": False,
        "callbacks": [],
        "deadline_monotonic": time.monotonic() + 15,
    })

    budgeted = [kwargs for kwargs in model.invoke_kwargs if kwargs]
    assert budgeted, "Expected at least one model call with a per-call timeout"
    for kwargs in budgeted:
        assert 10 < kwargs["timeout"] <= 15


def test_invoke_model_response_honours_zero_retries_under_tight_budget():
    from google.api_core.exceptions import ServiceUnavailable

    calls = []

    class FailingModel:
        def invoke(self, prompt, config=None, **kwargs):
            calls.append(1)
            raise ServiceUnavailable("temporary")

    with pytest.raises(ServiceUnavailable):
        nlq_graph._invoke_model_response(
            FailingModel(), "prompt", [], timeout=12.0, max_retries=0
        )
    assert len(calls) == 1


def test_nlq_restricted_intent_returns_restricted_status():
    from langchain_community.chat_models import FakeListChatModel

    fake_llm = FakeListChatModel(responses=["customers"])
    res = _make_graph(fake_llm).invoke({
        "question": "Cho tôi số điện thoại của khách hàng",
        "repair_count": 0,
        "repaired": False,
        "callbacks": [],
    })
    assert res.get("status") == "restricted"
    assert "bị hạn chế" in res["answer"]
    assert res["sql"] == ""


def test_sensitive_attribute_returns_restricted_without_fallback():
    """Ca 2: 'Email của khách hàng X' trả về status='restricted' và TUYỆT ĐỐI không fallback sang RAG."""
    from app.core.nlq_policy import has_sensitive_mention

    q_sensitive = "Email của khách hàng Nguyễn Văn A là gì?"
    assert has_sensitive_mention(q_sensitive) is True

    nlq_res = nlq_graph.ask_nlq(q_sensitive)
    assert nlq_res["status"] == "restricted"
    assert nlq_res["error_code"] == "policy_denied"
    assert "quyền riêng tư" in nlq_res["answer"]
    assert "hạng thành viên" in nlq_res["answer"]
    assert "phân khúc" not in nlq_res["answer"]


@pytest.mark.asyncio
async def test_orchestrator_blocks_sensitive_fallback_to_rag():
    """Đảm bảo Backend Orchestrator chặn không cho câu hỏi có cờ nhạy cảm fallback sang RAG."""
    from app.core.orchestrator import AskUnifiedRequest, ask_orchestrated

    req = AskUnifiedRequest(
        question="Số điện thoại của khách hàng là bao nhiêu?",
        mode="auto",
    )
    res = await ask_orchestrated(req)
    assert res.status == "restricted"
    assert res.error_code == "policy_denied"
    assert res.fallback_from is None


def test_typed_exceptions_policy_vs_schema():
    """Kiểm tra ngoại lệ có kiểu PolicyViolation vs SchemaMismatch (kế thừa ValueError)."""
    from app.core.nlq_policy import PolicyViolation, SchemaMismatch

    with pytest.raises(PolicyViolation) as exc_info:
        validate_sql("SELECT phone FROM customers", ALL_TABLES)
    assert issubclass(PolicyViolation, ValueError)
    assert "nhạy cảm" in str(exc_info.value)

    with pytest.raises(PolicyViolation):
        validate_sql("DELETE FROM orders", ALL_TABLES)

    with pytest.raises(SchemaMismatch) as exc_info:
        validate_sql("SELECT * FROM employees", ALL_TABLES)
    assert issubclass(SchemaMismatch, ValueError)
    assert "ngoài schema" in str(exc_info.value)


def test_api_ask_unified_endpoint():
    """Kiểm tra endpoint /api/ask phản hồi chuẩn JSON với header X-API-Key."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/ask",
        json={"question": "Xóa toàn bộ đơn hàng cũ", "mode": "auto"},
        headers={"X-API-Key": settings.app_api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "unsupported_action"
    assert data["status"] == "unsupported_action"
    assert "chỉ đọc" in data["answer"]
