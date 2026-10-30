"""LangGraph based, read-only natural language query workflow."""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlglot import exp, parse
from sqlglot.optimizer.scope import traverse_scope

from app.core.config import settings
from app.core.llm import extract_message_text, get_llm, get_model_provider
from app.core.nlq_schema import _connect, load_schema, schema_catalog, select_relevant_tables
from app.core.tracing import get_langfuse_handler, langfuse_observation, update_langfuse_observation

_SENSITIVE = re.compile(r"password|passwd|secret|token|email|phone|mobile|address|account_number|national_id|full_name", re.I)
_SENSITIVE_PROJECTION_RE = re.compile(
    r"\b(?:số\s+điện\s+thoại|so\s+dien\s+thoai|sđt|sdt|email|thư\s+điện\s+tử|thu\s+dien\s+tu|"
    r"mật\s+khẩu|mat\s+khau|địa\s+chỉ|dia\s+chi|cmnd|cccd|căn\s+cước|can\s+cuoc|"
    r"tài\s+khoản\s+ngân\s+hàng|tai\s+khoan\s+ngan\s+hang|password|phone)\b",
    re.IGNORECASE,
)
_AGGREGATE_COUNT_RE = re.compile(
    r"\b(?:bao\s+nhiêu|mấy|tổng\s+số|đếm|thống\s+kê|tỷ\s+lệ|số\s+lượng)\b",
    re.IGNORECASE,
)
_ALLOWED_FUNCTIONS = {
    "abs", "avg", "coalesce", "concat", "count", "date_part", "extract",
    "greatest", "least", "length", "lower", "max", "min", "nullif", "round",
    "sum", "timestamp_trunc", "upper", "to_char", "date_trunc", "time_to_str",
    "cast", "trim", "and", "or", "not", "date",
}
logger = logging.getLogger(__name__)


class NlqState(TypedDict, total=False):
    question: str
    schema: list[dict]
    selected_tables: list[dict]
    schema_text: str
    sql: str
    rows: list[dict]
    columns: list[str]
    answer: str
    truncated: bool
    out_of_scope: bool
    deadline_monotonic: float
    repair_count: int
    error: str
    repaired: bool
    model: Any
    callbacks: list[Any]
    restricted: bool
    suggested_columns: list[str]
    status: str


class NlqRequestTimeout(TimeoutError):
    """Raised when the NLQ graph exceeds its request time budget."""


def _asks_merchandise_revenue(question: str | None) -> bool:
    if not question:
        return False
    folded = question.casefold().replace("đ", "d")
    folded = "".join(
        char for char in unicodedata.normalize("NFKD", folded)
        if not unicodedata.combining(char)
    )
    return bool(re.search(r"\bdoanh\s+(?:thu|so)\b", folded))


def _is_retryable_provider_error(exc: Exception) -> bool:
    name = type(exc).__name__.casefold()
    if "timeout" in name or "deadline" in name:
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            code = None
    if status is None:
        status = getattr(code, "value", code)
    try:
        return int(status) in {429, 500, 502, 503, 504}
    except (TypeError, ValueError):
        return False


def _invoke_model_response(llm, prompt: str, callbacks: list[Any]):
    for attempt in range(settings.nlq_model_max_retries + 1):
        try:
            return llm.invoke(prompt, config={"callbacks": callbacks})
        except Exception as exc:
            if attempt >= settings.nlq_model_max_retries or not _is_retryable_provider_error(exc):
                raise
            logger.warning("Retrying transient NLQ provider error (%s)", type(exc).__name__)
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("NLQ model call did not return")


def _invoke_model(llm, prompt: str, callbacks: list[Any]) -> str:
    return extract_message_text(_invoke_model_response(llm, prompt, callbacks))


def _extract_sql(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    if text[:4].casefold() == "sql\n":
        text = text[4:].strip()
    return text


def _has_canceled_order_exclusion(root: exp.Select) -> bool:
    order_aliases = {
        table.alias_or_name.casefold()
        for table in root.find_all(exp.Table)
        if table.name.casefold() == "orders"
    }
    if not order_aliases:
        return False

    predicates = []
    where = root.args.get("where")
    if where is not None:
        predicates.extend(where.walk())
    for join in root.find_all(exp.Join):
        on_clause = join.args.get("on")
        if on_clause is not None:
            predicates.extend(on_clause.walk())

    def is_order_status(node: exp.Expression) -> bool:
        if not isinstance(node, exp.Column) or node.name.casefold() != "status":
            return False
        return not node.table or node.table.casefold() in order_aliases

    def is_canceled_literal(node: exp.Expression) -> bool:
        return (
            isinstance(node, exp.Literal)
            and node.is_string
            and str(node.this).casefold() == "đã hủy"
        )

    for predicate in predicates:
        if isinstance(predicate, exp.NEQ):
            if (
                is_order_status(predicate.this) and is_canceled_literal(predicate.expression)
            ) or (
                is_order_status(predicate.expression) and is_canceled_literal(predicate.this)
            ):
                return True
        if isinstance(predicate, exp.Not) and isinstance(predicate.this, exp.In):
            inclusion = predicate.this
            if is_order_status(inclusion.this) and any(
                is_canceled_literal(value) for value in inclusion.expressions
            ):
                return True
        if isinstance(predicate, exp.In) and not isinstance(getattr(predicate, "parent", None), exp.Not):
            if is_order_status(predicate.this):
                values = [
                    str(value.this).casefold()
                    for value in predicate.expressions
                    if isinstance(value, exp.Literal) and value.is_string
                ]
                # Whitelist of valid active statuses that safely excludes 'đã hủy'
                valid_active = {"mới", "đã xác nhận", "đang giao", "đã giao"}
                if values and "đã hủy" not in values and set(values).issubset(valid_active):
                    return True
    return False


def validate_sql(
    sql: str,
    allowed_tables: set[str],
    allowed_columns: dict[str, set[str]] | None = None,
    *,
    question: str | None = None,
) -> str:
    """Accept one safe SELECT and impose a hard row cap before execution."""
    statements = [node for node in parse(sql, read="postgres") if node is not None and not isinstance(node, exp.Semicolon)]
    if len(statements) != 1:
        raise ValueError("Chỉ cho phép một câu lệnh SQL.")
    root = statements[0]
    if not isinstance(root, exp.Select):
        raise ValueError("Chỉ cho phép truy vấn SELECT hoặc WITH ... SELECT.")
    if root.args.get("into") or root.args.get("locks"):
        raise ValueError("Không cho phép SELECT INTO hoặc khóa dòng.")
    for node in root.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Command, exp.Merge, exp.Transaction)):
            raise ValueError("Truy vấn chứa thao tác ghi hoặc lệnh không được phép.")
        if isinstance(node, exp.Func):
            function_name = (
                node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            ).casefold()
            if function_name not in _ALLOWED_FUNCTIONS:
                raise ValueError("Truy vấn gọi hàm ngoài danh sách cho phép.")
    cte_names = {cte.alias_or_name.casefold() for cte in root.find_all(exp.CTE)}
    table_or_alias_names = set()
    referenced = set()

    for scope in traverse_scope(root):
        for name, source in scope.sources.items():
            table_or_alias_names.add(name.casefold())
            if isinstance(source, exp.Table):
                table_or_alias_names.add(source.name.casefold())

        for table in scope.tables:
            if table.catalog:
                raise ValueError("Truy vấn tham chiếu schema hoặc catalog không được phép.")
            if table.db and table.db.casefold() not in ("", "public"):
                raise ValueError("Truy vấn tham chiếu schema hoặc catalog không được phép.")

            # Bảng có tiền tố schema (vd: public.rag_chunks) hoặc không trùng tên CTE là bảng thật
            is_physical = bool(table.db) or (table.name.casefold() not in cte_names)
            if is_physical:
                referenced.add(table.name.casefold())

    allowed = {name.casefold() for name in allowed_tables}
    if not referenced or not referenced.issubset(allowed):
        raise ValueError("Truy vấn tham chiếu bảng ngoài schema được chọn.")

    # Cấm chọn trực tiếp toàn bộ dòng qua alias bảng hoặc tên bảng (e.g. SELECT c FROM customers c)
    for scope in traverse_scope(root):
        for col in scope.columns:
            if not col.table and col.name.casefold() in table_or_alias_names:
                parent = col.parent
                while parent and parent != scope.expression:
                    if parent in scope.expression.selects:
                        raise ValueError("Không cho phép chọn trực tiếp cả dòng hoặc alias.")
                    parent = parent.parent
                if col in scope.expression.selects:
                    raise ValueError("Không cho phép chọn trực tiếp cả dòng hoặc alias.")

    for select in root.find_all(exp.Select):
        for expr in select.expressions:
            for col in expr.find_all(exp.Column):
                if _SENSITIVE.search(col.name):
                    raise ValueError(f"Truy vấn chọn cột nhạy cảm '{col.name}' và đã bị chặn.")

    if allowed_columns:
        allowed_column_names = {
            table.casefold(): {column.casefold() for column in columns}
            for table, columns in allowed_columns.items()
        }
        output_aliases = {
            projection.alias.casefold()
            for select in root.find_all(exp.Select)
            for projection in select.expressions
            if isinstance(projection, exp.Alias) and projection.alias
        }
        for select in root.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star) or (
                    isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
                ):
                    raise ValueError("Không cho phép SELECT *; hãy chọn cột cụ thể.")
        table_aliases = {}
        for scope in traverse_scope(root):
            for alias, source in scope.sources.items():
                if isinstance(source, exp.Table):
                    table_aliases[alias.casefold()] = source.name.casefold()
                elif hasattr(source, "alias_or_name"):
                    table_aliases[alias.casefold()] = source.alias_or_name.casefold()
        for name in referenced:
            table_aliases[name.casefold()] = name.casefold()

        for column in root.find_all(exp.Column):
            name = column.name.casefold()
            qualifier = column.table.casefold()
            if qualifier and qualifier in cte_names:
                continue
            if qualifier:
                table_name = table_aliases.get(qualifier)
                if table_name is None:
                    raise ValueError("Truy vấn tham chiếu alias bảng không được xác định.")
                if name not in allowed_column_names.get(table_name, set()):
                    raise ValueError("Truy vấn tham chiếu cột ngoài schema được chọn.")
            elif name not in output_aliases:
                if not any(name in allowed_column_names.get(table, set()) for table in referenced):
                    raise ValueError("Truy vấn tham chiếu cột ngoài schema được chọn.")
    uses_order_revenue = any(
        isinstance(node, exp.Sum)
        and any(
            isinstance(product, exp.Mul)
            and {column.name.casefold() for column in product.find_all(exp.Column)}
            >= {"quantity", "unit_price"}
            for product in node.find_all(exp.Mul)
        )
        for node in root.find_all(exp.Sum)
    )
    if uses_order_revenue and _asks_merchandise_revenue(question):
        if "orders" not in referenced:
            raise ValueError("Truy vấn doanh thu phải JOIN orders để loại đơn đã hủy.")
        if not _has_canceled_order_exclusion(root):
            raise ValueError("Truy vấn doanh thu phải loại orders.status = 'đã hủy'.")
    limit = root.args.get("limit")
    if limit is None or not isinstance(limit.expression, exp.Literal) or not limit.expression.is_int:
        # Fetch one sentinel row beyond the public cap so execution can report
        # whether the displayed result was truncated.
        root = root.limit(settings.nlq_max_rows + 1)
    elif int(limit.expression.this) > settings.nlq_max_rows:
        root = root.limit(settings.nlq_max_rows + 1)
    return root.sql(dialect="postgres")


def _run_readonly(sql: str) -> tuple[list[str], list[dict], bool]:
    with _connect() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SELECT set_config('statement_timeout', %s, true)", (f"{settings.nlq_statement_timeout_ms}ms",))
        cursor = conn.execute(sql)
        columns = [column.name for column in cursor.description]
        if any(_SENSITIVE.search(name) for name in columns):
            raise ValueError("Kết quả chứa cột nhạy cảm và đã bị chặn.")
        fetched = cursor.fetchmany(settings.nlq_max_rows + 1)
        truncated = len(fetched) > settings.nlq_max_rows
        rows = [dict(zip(columns, row)) for row in fetched[: settings.nlq_max_rows]]
    return columns, rows, truncated


def _make_graph(llm):
    def load(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-schema-introspection", input={"question": state["question"]}) as span:
            tables = load_schema()
            update_langfuse_observation(span, output={"table_count": len(tables)})
        return {"schema": tables}

    def link(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-schema-linking", input={"question": state["question"]}) as span:
            selected = select_relevant_tables(
                state["question"], state["schema"], llm=llm,
                callbacks=state.get("callbacks", []),
            )
            text = schema_catalog(selected)
            update_langfuse_observation(span, output={"selected_tables": [t["name"] for t in selected]})

        q = state["question"]
        is_sensitive = bool(_SENSITIVE_PROJECTION_RE.search(q))
        is_aggregate = bool(_AGGREGATE_COUNT_RE.search(q))
        if is_sensitive and not is_aggregate:
            selected_names = {t["name"] for t in selected}
            if "customers" in selected_names or any(w in q.lower() for w in ["khách", "khach"]):
                suggested_cols = []
                for t in state.get("schema", []):
                    if t["name"] == "customers":
                        suggested_cols.extend(c["name"] for c in t["columns"] if not _SENSITIVE.search(c["name"]))
                vi_labels = {
                    "customer_id": "mã khách hàng",
                    "name": "tên khách hàng",
                    "customer_segment": "phân khúc khách hàng",
                    "city": "thành phố",
                    "total_spend": "tổng chi tiêu",
                    "created_at": "ngày tạo",
                }
                display = [vi_labels.get(c, c) for c in suggested_cols if c in vi_labels]
                return {
                    "selected_tables": selected,
                    "schema_text": text,
                    "restricted": True,
                    "suggested_columns": display or ["tên khách hàng", "phân khúc", "thành phố", "tổng chi tiêu"],
                }

        return {"selected_tables": selected, "schema_text": text, "out_of_scope": not selected}

    def restricted(state: NlqState) -> NlqState:
        cols = ", ".join(state.get("suggested_columns", ["tên khách hàng", "phân khúc", "thành phố", "tổng chi tiêu"]))
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "answer": f"Thông tin liên hệ khách hàng (số điện thoại, email, địa chỉ) bị hạn chế để bảo vệ quyền riêng tư. Bạn có thể tra cứu các trường thông tin được phép như: {cols}.",
            "repaired": False,
            "truncated": False,
            "error": "",
            "status": "restricted",
        }

    def out_of_scope(state: NlqState) -> NlqState:
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "answer": "Câu hỏi không liên quan đến dữ liệu bán hàng có sẵn.",
            "repaired": False,
            "truncated": False,
            "error": "",
            "status": "out_of_scope",
        }

    def generate(state: NlqState) -> NlqState:
        prompt = (
            "Bạn là bộ sinh SQL PostgreSQL. Hãy tạo đúng một câu SELECT để trả lời câu hỏi. "
            "Chỉ dùng bảng/cột trong schema; không lấy thông tin cá nhân/nhạy cảm; không giải thích. "
            "Luôn đặt LIMIT phù hợp. Chỉ khi câu hỏi hỏi doanh thu/doanh số hàng hóa, dùng "
            "SUM(order_items.quantity * order_items.unit_price), JOIN orders với order_items và loại "
            "đơn status='đã hủy'; doanh thu gồm các trạng thái mới, đã xác nhận, đang giao, đã giao. "
            "Không tự áp dụng điều kiện loại đơn hủy cho câu hỏi về tổng giá trị hàng đã đặt hoặc "
            "giá trị trung bình đơn hàng; ở các câu đó hãy giữ đúng phạm vi câu hỏi. Không dùng "
            "shipping_fee làm doanh thu hàng hóa. "
            "Ví dụ khi schema có orders/order_items: SELECT date_trunc('month', orders.order_date) AS month, "
            "SUM(order_items.quantity * order_items.unit_price) AS revenue FROM orders "
            "JOIN order_items ON order_items.order_id = orders.order_id WHERE orders.status <> 'đã hủy' "
            "GROUP BY month ORDER BY month LIMIT 100.\n"
            "SCHEMA:\n" + state["schema_text"] +
            "\nQUESTION:\n" + state["question"]
        )
        with langfuse_observation(name="nlq-sql-generation", input={"question": state["question"], "schema": state["schema_text"]}) as span:
            sql = _extract_sql(_invoke_model(llm, prompt, state.get("callbacks", [])))
            update_langfuse_observation(span, output={"sql": sql})
        return {"sql": sql, "error": ""}

    def validate(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-sql-validation", input={"sql": state["sql"]}) as span:
            try:
                safe_sql = validate_sql(
                    state["sql"],
                    {t["name"] for t in state["selected_tables"]},
                    question=state["question"],
                )
                update_langfuse_observation(span, output={"accepted": True, "sql": safe_sql})
                return {"sql": safe_sql, "error": ""}
            except Exception as exc:
                update_langfuse_observation(span, output={"accepted": False, "reason": str(exc)}, level="WARNING")
                return {"error": str(exc)}

    def execute(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-sql-execution", input={"sql": state["sql"]}) as span:
            try:
                columns, rows, truncated = _run_readonly(state["sql"])
                update_langfuse_observation(
                    span,
                    output={"row_count": len(rows), "columns": columns, "truncated": truncated},
                )
                return {"columns": columns, "rows": rows, "truncated": truncated, "error": ""}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:1000]
                update_langfuse_observation(span, output={"error": error}, level="ERROR")
                return {"error": error}

    def repair(state: NlqState) -> NlqState:
        count = state.get("repair_count", 0) + 1
        revenue_rule = (
            "Vì câu hỏi hỏi doanh thu/doanh số hàng hóa, bắt buộc loại đơn status='đã hủy'. "
            if _asks_merchandise_revenue(state["question"])
            else "Không tự loại đơn hủy trừ khi câu hỏi yêu cầu; giữ đúng phạm vi câu hỏi. "
        )
        prompt = (
            "Sửa câu SQL PostgreSQL để chỉ còn một truy vấn SELECT hợp lệ. "
            "Chỉ dùng schema đã cung cấp. " + revenue_rule +
            "Không tính shipping_fee là doanh thu hàng hóa. Trả đúng SQL, không giải thích.\nSCHEMA:\n" + state["schema_text"] +
            "\nQUESTION:\n" + state["question"] + "\nSQL lỗi:\n" + state["sql"] +
            "\nLỗi kiểm tra/thực thi:\n" + state["error"]
        )
        with langfuse_observation(name="nlq-sql-repair", input={"attempt": count, "error": state["error"]}) as span:
            sql = _extract_sql(_invoke_model(llm, prompt, state.get("callbacks", [])))
            update_langfuse_observation(span, output={"sql": sql})
        return {"sql": sql, "repair_count": count, "repaired": True, "error": ""}

    _SAMPLE_ROWS = 25

    def interpret(state: NlqState) -> NlqState:
        rows = state.get("rows", [])
        sample_rows = rows[:_SAMPLE_ROWS]
        is_partial = state.get("truncated") or len(rows) > _SAMPLE_ROWS
        partial_note = (
            f"\nLƯU Ý: Kết quả hiển thị mẫu {len(sample_rows)} trong tổng số {len(rows)} dòng lấy về; "
            "không được tự suy ra tổng số hoặc kết luận tuyệt đối từ mẫu này."
            if is_partial
            else ""
        )
        prompt = (
            "Trả lời ngắn gọn bằng tiếng Việt, dựa đúng vào kết quả SQL dưới đây. "
            "Nếu không có dòng nào, nói rõ không tìm thấy dữ liệu.\nCÂU HỎI:\n" + state["question"] +
            "\nCỘT:\n" + json.dumps(state.get("columns", []), ensure_ascii=False) +
            "\nKẾT QUẢ:\n" + json.dumps(sample_rows, ensure_ascii=False, default=str) +
            partial_note
        )
        with langfuse_observation(name="nlq-answer-interpretation", input={"row_count": len(rows)}) as span:
            answer = _invoke_model(llm, prompt, state.get("callbacks", []))
            update_langfuse_observation(span, output={"answer": answer})
        return {"answer": answer, "status": "ok"}

    builder = StateGraph(NlqState)
    def timed_node(name, fn):
        def run(state):
            started = time.perf_counter()
            deadline = state.get("deadline_monotonic")
            if deadline is not None and time.monotonic() >= deadline:
                raise NlqRequestTimeout("NLQ request deadline exceeded before " + name)
            logger.info("NLQ stage started: %s", name)
            try:
                result = fn(state)
            except Exception as exc:
                logger.warning(
                    "NLQ stage failed: %s after %.2fs (%s)",
                    name, time.perf_counter() - started, type(exc).__name__,
                )
                raise
            if deadline is not None and time.monotonic() >= deadline:
                raise NlqRequestTimeout("NLQ request deadline exceeded during " + name)
            logger.info("NLQ stage completed: %s in %.2fs", name, time.perf_counter() - started)
            return result
        return run

    builder.add_node("load_schema", timed_node("load_schema", load))
    builder.add_node("schema_link", timed_node("schema_link", link))
    builder.add_node("restricted", timed_node("restricted", restricted))
    builder.add_node("out_of_scope", timed_node("out_of_scope", out_of_scope))
    builder.add_node("generate_sql", timed_node("generate_sql", generate))
    builder.add_node("validate_sql", timed_node("validate_sql", validate))
    builder.add_node("execute_sql", timed_node("execute_sql", execute))
    builder.add_node("repair_sql", timed_node("repair_sql", repair))
    builder.add_node("interpret", timed_node("interpret", interpret))
    builder.add_edge(START, "load_schema")
    builder.add_edge("load_schema", "schema_link")

    def after_schema_link(state):
        if state.get("restricted"):
            return "restricted"
        if state.get("out_of_scope"):
            return "out_of_scope"
        return "generate_sql"

    builder.add_conditional_edges(
        "schema_link",
        after_schema_link,
        {"restricted": "restricted", "out_of_scope": "out_of_scope", "generate_sql": "generate_sql"},
    )
    builder.add_edge("restricted", END)
    builder.add_edge("out_of_scope", END)
    builder.add_edge("generate_sql", "validate_sql")

    def after_validation(state):
        if state.get("error"):
            return "repair_sql" if state.get("repair_count", 0) < settings.nlq_max_repairs else END
        return "execute_sql"

    def after_execution(state):
        if state.get("error"):
            # Lỗi mất kết nối CSDL (OperationalError, timeout vật lý...) thì kết thúc ngay, không repair
            if "OperationalError" in state["error"] or "database_unavailable" in state["error"].lower():
                return END
            return "repair_sql" if state.get("repair_count", 0) < settings.nlq_max_repairs else END
        return "interpret"

    builder.add_conditional_edges("validate_sql", after_validation, {"repair_sql": "repair_sql", "execute_sql": "execute_sql", END: END})
    builder.add_conditional_edges("execute_sql", after_execution, {"repair_sql": "repair_sql", "interpret": "interpret", END: END})
    builder.add_edge("repair_sql", "validate_sql")
    builder.add_edge("interpret", END)
    return builder.compile()


def ask_nlq(question: str, *, session_id: str | None = None, user_id: str | None = None) -> dict:
    model_options = (
        {"thinking_config": {"thinking_level": "low"}}
        if get_model_provider(settings.nlq_model) == "google_genai"
        else {}
    )
    max_graph_model_calls = settings.nlq_max_repairs + 3  # link, generate, repairs, interpret
    deadline_seconds = getattr(settings, "nlq_request_deadline_seconds", 60.0)
    # Google GenAI quy định deadline tối thiểu phải >= 10s. Tránh chia max_attempts làm timeout < 10s.
    per_call_budget = max(10.0, deadline_seconds / max(1, max_graph_model_calls))
    model_timeout = max(10.0, min(settings.nlq_model_timeout_seconds, per_call_budget))
    deadline = time.monotonic() + deadline_seconds
    llm = get_llm(
        settings.nlq_model,
        temperature=0,
        timeout=model_timeout,
        max_retries=settings.nlq_model_max_retries,
        **model_options,
    )
    handler = get_langfuse_handler()
    callbacks = [handler] if handler is not None else []
    from app.core.tracing import langfuse_context
    with langfuse_context(session_id=session_id, user_id=user_id, tags=["nlq", "api"], metadata={"model_role": settings.nlq_model}):
        started = time.perf_counter()
        try:
            result = _make_graph(llm).invoke({
                "question": question,
                "repair_count": 0,
                "repaired": False,
                "callbacks": callbacks,
                "deadline_monotonic": deadline,
            }, config={"callbacks": callbacks})
            if time.monotonic() >= deadline:
                raise NlqRequestTimeout("NLQ request deadline exceeded.")
        except Exception as exc:
            logger.warning(
                "NLQ request failed after %.2fs (%s)",
                time.perf_counter() - started, type(exc).__name__,
            )
            raise
        logger.info("NLQ request completed in %.2fs", time.perf_counter() - started)
    if result.get("error"):
        raise ValueError(f"Không thể chạy truy vấn an toàn: {result['error']}")
    return {
        "sql": result.get("sql", ""),
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "answer": result.get("answer", "Không thể diễn giải kết quả."),
        "repaired": result.get("repaired", False),
        "truncated": result.get("truncated", False),
        "status": result.get("status", "ok"),
    }
