"""LangGraph based, read-only natural language query workflow."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, TypedDict

import psycopg
from psycopg_pool import PoolTimeout
from langgraph.graph import END, START, StateGraph
from sqlglot import exp, parse

from app.core.config import settings
from app.core.llm import extract_message_text, get_llm, get_model_provider
from app.core.nlq_schema import _SENSITIVE, _connect, load_schema, schema_catalog, select_relevant_tables
from app.core.tracing import get_langfuse_handler, langfuse_observation, update_langfuse_observation

_ALLOWED_FUNCTIONS = {
    "abs", "avg", "coalesce", "concat", "count", "date_part", "extract",
    "greatest", "least", "length", "lower", "max", "min", "nullif", "round",
    "sum", "timestamp_trunc", "upper",
}
logger = logging.getLogger(__name__)


class NlqUnsafeQueryError(ValueError):
    """The generated statement could not pass the NLQ read-only guard."""


class NlqExecutionError(ValueError):
    """A safe query could not be executed for this request."""


class NlqDatabaseUnavailable(RuntimeError):
    """The NLQ database is not reachable or is not configured."""


class NlqState(TypedDict, total=False):
    question: str
    schema: list[dict]
    selected_tables: list[dict]
    schema_text: str
    sql: str
    rows: list[dict]
    columns: list[str]
    answer: str
    repair_count: int
    error: str
    repaired: bool
    truncated: bool
    out_of_scope: bool
    error_stage: str
    callbacks: list[Any]


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


def validate_sql(
    sql: str,
    allowed_tables: set[str],
    allowed_columns: dict[str, set[str]] | None = None,
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
    source_tables = [
        table for table in root.find_all(exp.Table)
        if table.name.casefold() not in cte_names
    ]
    referenced = {table.name.casefold() for table in source_tables}
    allowed = {name.casefold() for name in allowed_tables}
    if not referenced or not referenced.issubset(allowed):
        raise ValueError("Truy vấn tham chiếu bảng ngoài schema được chọn.")
    table_aliases = {
        table.alias_or_name.casefold(): table.name.casefold()
        for table in source_tables
    }
    table_aliases.update({name: name for name in referenced})
    allowed_column_names = {
        table.casefold(): {column.casefold() for column in columns}
        for table, columns in (allowed_columns or {}).items()
    }
    output_aliases = {
        projection.alias_or_name.casefold()
        for select in root.find_all(exp.Select)
        for projection in select.expressions
        if projection.alias_or_name
    }
    for select in root.find_all(exp.Select):
        for projection in select.expressions:
            if isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
            ):
                raise ValueError("Không cho phép SELECT *; hãy chọn cột cụ thể.")
    for column in root.find_all(exp.Column):
        name = column.name.casefold()
        if _SENSITIVE.search(name):
            raise ValueError("Truy vấn tham chiếu cột nhạy cảm.")
        qualifier = column.table.casefold()
        if qualifier and qualifier in cte_names:
            continue
        if qualifier:
            table_name = table_aliases.get(qualifier)
            if table_name is None:
                raise ValueError("Truy vấn tham chiếu alias bảng không được xác định.")
            if allowed_column_names and name not in allowed_column_names.get(table_name, set()):
                raise ValueError("Truy vấn tham chiếu cột ngoài schema được chọn.")
        elif name not in output_aliases and allowed_column_names:
            if not any(name in allowed_column_names.get(table, set()) for table in referenced):
                raise ValueError("Truy vấn tham chiếu cột ngoài schema được chọn.")
    limit = root.args.get("limit")
    if limit is None or not isinstance(limit.expression, exp.Literal) or not limit.expression.is_int:
        root = root.limit(settings.nlq_max_rows + 1)
    elif int(limit.expression.this) >= settings.nlq_max_rows:
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


def _safe_execution_error(exc: Exception) -> str:
    """Avoid forwarding PostgreSQL error text that may include a bound value."""
    if isinstance(exc, psycopg.Error):
        sqlstate = getattr(exc, "sqlstate", None)
        return f"PostgreSQL execution failed (SQLSTATE {sqlstate})." if sqlstate else "PostgreSQL execution failed."
    return f"{type(exc).__name__}: truy vấn không thể thực thi."


def _make_graph(llm):
    def load(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-schema-introspection", input={"question": state["question"]}) as span:
            tables = load_schema()
            update_langfuse_observation(span, output={"table_count": len(tables)})
        return {"schema": tables}

    def link(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-scope-check", input={"question": state["question"]}) as span:
            relevant = select_relevant_tables(state["question"], state["schema"])
            # Keep lexical matching only as a cheap scope check. The schema has
            # nine small tables, so give SQL generation the complete catalog.
            update_langfuse_observation(
                span,
                output={"scope_match": bool(relevant), "schema_table_count": len(state["schema"])},
            )
        if not relevant:
            return {
                "selected_tables": [],
                "schema_text": "",
                "out_of_scope": True,
                "answer": "Câu hỏi chưa khớp với dữ liệu hiện có; hãy hỏi về đơn hàng, khách hàng, sản phẩm, kho hoặc tồn kho.",
            }
        return {
            "selected_tables": state["schema"],
            "schema_text": schema_catalog(state["schema"]),
            "out_of_scope": False,
        }

    def generate(state: NlqState) -> NlqState:
        prompt = (
            "Bạn là bộ sinh SQL PostgreSQL. Hãy tạo đúng một câu SELECT để trả lời câu hỏi. "
            "Chỉ dùng bảng/cột trong schema; không lấy thông tin cá nhân/nhạy cảm; không giải thích. "
            "Luôn đặt LIMIT phù hợp. Với doanh thu đơn hàng, dùng SUM(order_items.quantity * "
            "order_items.unit_price) và JOIN orders với order_items; không dùng shipping_fee làm doanh thu. "
            "Ví dụ khi schema có orders/order_items: SELECT date_trunc('month', orders.order_date) AS month, "
            "SUM(order_items.quantity * order_items.unit_price) AS revenue FROM orders "
            "JOIN order_items ON order_items.order_id = orders.order_id GROUP BY month ORDER BY month LIMIT 100.\n"
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
                    {
                        table["name"]: {column["name"] for column in table["columns"]}
                        for table in state["selected_tables"]
                    },
                )
                update_langfuse_observation(span, output={"accepted": True, "sql": safe_sql})
                return {"sql": safe_sql, "error": "", "error_stage": ""}
            except Exception as exc:
                update_langfuse_observation(span, output={"accepted": False, "reason": str(exc)}, level="WARNING")
                return {"error": str(exc), "error_stage": "validation"}

    def execute(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-sql-execution", input={"sql": state["sql"]}) as span:
            try:
                columns, rows, truncated = _run_readonly(state["sql"])
                update_langfuse_observation(span, output={"row_count": len(rows), "columns": columns})
                return {
                    "columns": columns, "rows": rows, "truncated": truncated,
                    "row_count": len(rows), "error": "", "error_stage": "",
                }
            except Exception as exc:
                error = _safe_execution_error(exc)
                update_langfuse_observation(span, output={"error": error}, level="ERROR")
                error_stage = (
                    "database_unavailable"
                    if isinstance(exc, (psycopg.OperationalError, PoolTimeout))
                    else "execution"
                )
                return {"error": error, "error_stage": error_stage}

    def repair(state: NlqState) -> NlqState:
        count = state.get("repair_count", 0) + 1
        prompt = (
            "Sửa câu SQL PostgreSQL để chỉ còn một truy vấn SELECT hợp lệ. "
            "Chỉ dùng schema đã cung cấp. Trả đúng SQL, không giải thích.\nSCHEMA:\n" + state["schema_text"] +
            "\nQUESTION:\n" + state["question"] + "\nSQL lỗi:\n" + state["sql"] +
            "\nLỗi kiểm tra/thực thi:\n" + state["error"]
        )
        with langfuse_observation(name="nlq-sql-repair", input={"attempt": count, "error": state["error"]}) as span:
            sql = _extract_sql(_invoke_model(llm, prompt, state.get("callbacks", [])))
            update_langfuse_observation(span, output={"sql": sql})
        return {"sql": sql, "repair_count": count, "repaired": True, "error": ""}

    def interpret(state: NlqState) -> NlqState:
        prompt = (
            "Trả lời ngắn gọn bằng tiếng Việt, dựa đúng vào kết quả SQL dưới đây. "
            "Nếu không có dòng nào, nói rõ không tìm thấy dữ liệu.\nCÂU HỎI:\n" + state["question"] +
            "\nCỘT:\n" + json.dumps(state.get("columns", []), ensure_ascii=False) +
            "\nMẪU TỐI ĐA 25 DÒNG ĐẦU:\n" + json.dumps(state.get("rows", [])[:25], ensure_ascii=False, default=str) +
            f"\nSỐ DÒNG TRẢ VỀ: {len(state.get('rows', []))}" +
            ("\nLƯU Ý: Kết quả truy vấn đã bị giới hạn; mẫu trên chỉ gồm tối đa 25 dòng đầu, không được suy ra tổng từ mẫu."
             if state.get("truncated") else "")
        )
        with langfuse_observation(name="nlq-answer-interpretation", input={"row_count": len(state.get("rows", []))}) as span:
            answer = _invoke_model(llm, prompt, state.get("callbacks", []))
            update_langfuse_observation(span, output={"answer": answer})
        return {"answer": answer}

    builder = StateGraph(NlqState)
    def timed_node(name, fn):
        def run(state):
            started = time.perf_counter()
            logger.info("NLQ stage started: %s", name)
            try:
                result = fn(state)
            except Exception as exc:
                logger.warning(
                    "NLQ stage failed: %s after %.2fs (%s)",
                    name, time.perf_counter() - started, type(exc).__name__,
                )
                raise
            logger.info("NLQ stage completed: %s in %.2fs", name, time.perf_counter() - started)
            return result
        return run

    builder.add_node("load_schema", timed_node("load_schema", load))
    builder.add_node("schema_link", timed_node("schema_link", link))
    builder.add_node("generate_sql", timed_node("generate_sql", generate))
    builder.add_node("validate_sql", timed_node("validate_sql", validate))
    builder.add_node("execute_sql", timed_node("execute_sql", execute))
    builder.add_node("repair_sql", timed_node("repair_sql", repair))
    builder.add_node("interpret", timed_node("interpret", interpret))
    builder.add_edge(START, "load_schema")
    builder.add_edge("load_schema", "schema_link")
    def after_link(state):
        return END if state.get("out_of_scope") else "generate_sql"

    builder.add_conditional_edges("schema_link", after_link, {"generate_sql": "generate_sql", END: END})
    builder.add_edge("generate_sql", "validate_sql")

    def after_validation(state):
        if state.get("error"):
            return "repair_sql" if state.get("repair_count", 0) < settings.nlq_max_repairs else END
        return "execute_sql"

    def after_execution(state):
        if state.get("error"):
            return "repair_sql" if state.get("repair_count", 0) < settings.nlq_max_repairs else END
        return "interpret"

    builder.add_conditional_edges("validate_sql", after_validation, {"repair_sql": "repair_sql", "execute_sql": "execute_sql", END: END})
    builder.add_conditional_edges("execute_sql", after_execution, {"repair_sql": "repair_sql", "interpret": "interpret", END: END})
    builder.add_edge("repair_sql", "validate_sql")
    builder.add_edge("interpret", END)
    return builder.compile()


def ask_nlq(question: str, *, session_id: str | None = None) -> dict:
    model_options = (
        {"thinking_config": {"thinking_level": "low"}}
        if get_model_provider(settings.nlq_model) == "google_genai"
        else {}
    )
    llm = get_llm(
        settings.nlq_model,
        temperature=0,
        timeout=settings.nlq_model_timeout_seconds,
        # Provider SDK retry policies may retry timeouts too; transient HTTP
        # 429/5xx retries are handled narrowly in _invoke_model instead.
        max_retries=0,
        **({"num_ctx": settings.rag_num_ctx} if get_model_provider(settings.nlq_model) == "ollama" else {}),
        **model_options,
    )
    handler = get_langfuse_handler()
    callbacks = [handler] if handler is not None else []
    from app.core.tracing import langfuse_context
    # The API does not authenticate a person identity; never trust a client
    # supplied user_id as an identity attribute in Langfuse.
    with langfuse_context(session_id=session_id, tags=["nlq", "api"], metadata={"model_role": settings.nlq_model}):
        started = time.perf_counter()
        try:
            result = _make_graph(llm).invoke({
                "question": question,
                "repair_count": 0,
                "repaired": False,
                "callbacks": callbacks,
            }, config={"callbacks": callbacks})
        except Exception as exc:
            logger.warning(
                "NLQ request failed after %.2fs (%s)",
                time.perf_counter() - started, type(exc).__name__,
            )
            raise
        logger.info("NLQ request completed in %.2fs", time.perf_counter() - started)
    if result.get("error"):
        if result.get("error_stage") == "validation":
            raise NlqUnsafeQueryError("Không thể tạo câu SQL an toàn cho câu hỏi này.")
        if result.get("error_stage") == "database_unavailable":
            raise NlqDatabaseUnavailable("NLQ database is not reachable.")
        raise NlqExecutionError("Không thể thực thi câu truy vấn cho yêu cầu này.")
    if result.get("out_of_scope"):
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "answer": result["answer"],
            "repaired": False,
            "truncated": False,
        }
    return {
        "sql": result["sql"],
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "answer": result.get("answer", "Không thể diễn giải kết quả."),
        "repaired": result.get("repaired", False),
        "truncated": result.get("truncated", False),
    }
