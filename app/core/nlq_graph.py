"""LangGraph based, read-only natural language query workflow."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypedDict

import psycopg
from langgraph.graph import END, START, StateGraph
from sqlglot import exp, parse

from app.core.config import settings
from app.core.llm import extract_message_text, get_llm, get_model_provider
from app.core.nlq_schema import _connect, load_schema, schema_catalog, select_relevant_tables
from app.core.tracing import get_langfuse_handler, langfuse_observation, update_langfuse_observation

_SENSITIVE = re.compile(r"password|passwd|secret|token|email|phone|mobile|address|account_number|national_id|full_name", re.I)
_FORBIDDEN_FUNCTIONS = {"pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "dblink", "lo_export", "lo_import"}
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
    repair_count: int
    error: str
    repaired: bool
    model: Any
    callbacks: list[Any]


def _invoke_model(llm, prompt: str, callbacks: list[Any]) -> str:
    result = llm.invoke(prompt, config={"callbacks": callbacks})
    return extract_message_text(result)


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


def validate_sql(sql: str, allowed_tables: set[str]) -> str:
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
            function_name = (getattr(node, "name", "") or node.sql_name()).casefold()
            if function_name in _FORBIDDEN_FUNCTIONS:
                raise ValueError("Truy vấn gọi hàm hệ thống không được phép.")
    cte_names = {cte.alias_or_name.casefold() for cte in root.find_all(exp.CTE)}
    referenced = {table.name.casefold() for table in root.find_all(exp.Table) if table.name.casefold() not in cte_names}
    allowed = {name.casefold() for name in allowed_tables}
    if not referenced or not referenced.issubset(allowed):
        raise ValueError("Truy vấn tham chiếu bảng ngoài schema được chọn.")
    limit = root.args.get("limit")
    if limit is None or not isinstance(limit.expression, exp.Literal) or not limit.expression.is_int:
        root = root.limit(settings.nlq_max_rows)
    elif int(limit.expression.this) > settings.nlq_max_rows:
        root = root.limit(settings.nlq_max_rows)
    return root.sql(dialect="postgres")


def _run_readonly(sql: str) -> tuple[list[str], list[dict]]:
    with _connect() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SELECT set_config('statement_timeout', %s, true)", (f"{settings.nlq_statement_timeout_ms}ms",))
        cursor = conn.execute(sql)
        columns = [column.name for column in cursor.description]
        if any(_SENSITIVE.search(name) for name in columns):
            raise ValueError("Kết quả chứa cột nhạy cảm và đã bị chặn.")
        rows = [dict(zip(columns, row)) for row in cursor.fetchmany(settings.nlq_max_rows + 1)]
    return columns, rows[: settings.nlq_max_rows]


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
        return {"selected_tables": selected, "schema_text": text}

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
                safe_sql = validate_sql(state["sql"], {t["name"] for t in state["selected_tables"]})
                update_langfuse_observation(span, output={"accepted": True, "sql": safe_sql})
                return {"sql": safe_sql, "error": ""}
            except Exception as exc:
                update_langfuse_observation(span, output={"accepted": False, "reason": str(exc)}, level="WARNING")
                return {"error": str(exc)}

    def execute(state: NlqState) -> NlqState:
        with langfuse_observation(name="nlq-sql-execution", input={"sql": state["sql"]}) as span:
            try:
                columns, rows = _run_readonly(state["sql"])
                update_langfuse_observation(span, output={"row_count": len(rows), "columns": columns})
                return {"columns": columns, "rows": rows, "error": ""}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:1000]
                update_langfuse_observation(span, output={"error": error}, level="ERROR")
                return {"error": error}

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
            "\nKẾT QUẢ:\n" + json.dumps(state.get("rows", []), ensure_ascii=False, default=str)
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
    builder.add_edge("schema_link", "generate_sql")
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


def ask_nlq(question: str, *, session_id: str | None = None, user_id: str | None = None) -> dict:
    model_options = (
        {"thinking_config": {"thinking_level": "low"}}
        if get_model_provider(settings.nlq_model) == "google_genai"
        else {}
    )
    llm = get_llm(
        settings.nlq_model,
        temperature=0,
        timeout=settings.nlq_model_timeout_seconds,
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
            }, config={"callbacks": callbacks})
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
        "sql": result["sql"],
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "answer": result.get("answer", "Không thể diễn giải kết quả."),
        "repaired": result.get("repaired", False),
    }
