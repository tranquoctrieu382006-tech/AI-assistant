"""Evaluate model-generated SQL against the NLQ golden set, read-only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import uuid

from langfuse import Evaluation, get_client
import psycopg
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from app.core.config import PROJECT_ROOT, settings
from app.core.llm import MODEL_REGISTRY, extract_message_text, get_llm, get_model_provider
from app.core.nlq_graph import _extract_sql, _invoke_model_response, validate_sql
from app.core.nlq_schema import _SENSITIVE
from app.core.tracing import (
    flush_langfuse,
    get_langfuse_handler,
    init_tracing,
    is_langfuse_configured,
    langfuse_observation,
    update_langfuse_observation,
)
from scripts.evaluate_nlq_golden import results_match

LANGFUSE_DATASET_NAME = "ai-assistant-nlq-golden-v1"
MODEL_PRICING_USD_PER_MILLION = {
    # Gemini standard text pricing effective through 2026-12-31.
    "google_genai/gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "google_genai/gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "google_genai/gemini-3.6-flash": {"input": 0.75, "output": 3.75},
}


def _sql_features(sql: str) -> tuple[int, bool]:
    """Return parsed statement count and whether the outer query orders rows."""
    statements = [
        statement for statement in parse(sql, read="postgres")
        if statement is not None and not isinstance(statement, exp.Semicolon)
    ]
    has_order_by = len(statements) == 1 and statements[0].args.get("order") is not None
    return len(statements), has_order_by


def _split_ddl_definitions(body: str) -> list[str]:
    definitions = []
    start = depth = 0
    quote = None
    for index, char in enumerate(body):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            definitions.append(body[start:index])
            start = index + 1
    definitions.append(body[start:])
    return definitions


def _seed_schema_allowlist(schema: str) -> tuple[set[str], dict[str, set[str]]]:
    """Extract table/column names from seed DDL for the production SQL guard."""
    tables: set[str] = set()
    columns: dict[str, set[str]] = {}
    for match in re.finditer(r"CREATE\s+TABLE\s+([\w\"]+)\s*\((.*?)\)\s*;", schema, re.I | re.S):
        table = match.group(1).strip('"')
        tables.add(table)
        columns[table] = set()
        for definition in _split_ddl_definitions(match.group(2)):
            tokens = definition.strip().split(None, 1)
            if not tokens:
                continue
            name = tokens[0].strip('"')
            if name.casefold() in {"constraint", "primary", "foreign", "unique", "check", "exclude"}:
                continue
            if not _SENSITIVE.search(name):
                columns[table].add(name)
    return tables, columns


def _response_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    fields = {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "total_tokens": usage.get("total_tokens"),
    }
    if fields["total_tokens"] is None and fields["input_tokens"] is not None and fields["output_tokens"] is not None:
        fields["total_tokens"] = fields["input_tokens"] + fields["output_tokens"]
    return {key: value for key, value in fields.items() if isinstance(value, (int, float))}


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


def _relaxed_execution_results_match(
    actual: list[dict], expected: list[dict], *, ordered: bool = False,
) -> bool:
    """Allow extra projected columns, but require every golden value to be present.

    Column aliases/order are ignored. Numeric values use a small tolerance and
    month labels such as ``1``/``2026-01`` are normalized. Missing golden values
    still fail, so omitting a requested metric is not scored as correct.
    """
    from datetime import date, datetime
    from decimal import Decimal, InvalidOperation

    def equivalent(left, right):
        if left == right:
            return True
        left_text, right_text = str(left), str(right)
        if re.fullmatch(r"\d{1,2}", left_text) and re.fullmatch(r"\d{4}-\d{2}", right_text):
            return int(left_text) == int(right_text[-2:])
        if re.fullmatch(r"\d{4}-\d{2}", left_text) and re.fullmatch(r"\d{1,2}", right_text):
            return int(left_text[-2:]) == int(right_text)
        try:
            a, b = Decimal(left_text), Decimal(right_text)
            return abs(a - b) <= max(Decimal("0.01"), abs(b) * Decimal("1e-9"))
        except (InvalidOperation, ValueError):
            return False

    def row_matches(actual_row, expected_row):
        remaining = list(actual_row.values())
        for expected_value in expected_row.values():
            match = next((i for i, value in enumerate(remaining) if equivalent(value, expected_value)), None)
            if match is None:
                return False
            remaining.pop(match)
        return True

    if len(actual) != len(expected):
        return False
    if ordered:
        return all(row_matches(a, e) for a, e in zip(actual, expected))
    unmatched = list(actual)
    for expected_row in expected:
        match = next((i for i, row in enumerate(unmatched) if row_matches(row, expected_row)), None)
        if match is None:
            return False
        unmatched.pop(match)
    return True


def _load_schema_ddl() -> str:
    """Use seed DDL while omitting columns the production guard forbids."""
    seed = (PROJECT_ROOT / "data/demo_sales_warehouse/schema_seed.sql").read_text(encoding="utf-8-sig")
    statements = []
    for match in re.finditer(r"CREATE\s+TABLE\s+([\w\"]+)\s*\((.*?)\)\s*;", seed, re.I | re.S):
        definitions = _split_ddl_definitions(match.group(2))
        safe_definitions = []
        for definition in definitions:
            name = definition.strip().split(None, 1)[0].strip('"')
            is_constraint = name.casefold() in {"constraint", "primary", "foreign", "unique", "check", "exclude"}
            if (is_constraint and not _SENSITIVE.search(definition)) or (
                not is_constraint and not _SENSITIVE.search(name)
            ):
                safe_definitions.append(definition.strip())
        statements.append(f"CREATE TABLE {match.group(1)} ({', '.join(safe_definitions)});")
    if not statements:
        raise ValueError("No CREATE TABLE statements found in data/demo_sales_warehouse/schema_seed.sql")
    return "\n".join(statements)


def evaluate(
    model_role: str = "local",
    limit: int | None = None,
    *,
    trace_langfuse: bool = False,
    request_interval_seconds: float | None = None,
) -> dict:
    if not settings.nlq_database_url.strip():
        raise RuntimeError("NLQ_DATABASE_URL is required; set it to the PostgreSQL read-only account.")
    if trace_langfuse and not is_langfuse_configured():
        raise RuntimeError("Langfuse Cloud tracing was requested but is not configured.")
    if trace_langfuse:
        init_tracing()
    cases = json.loads((PROJECT_ROOT / "data/nlq_golden_set.json").read_text(encoding="utf-8"))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than zero.")
        cases = cases[:limit]
    schema = _load_schema_ddl()
    allowed_tables, allowed_columns = _seed_schema_allowlist(schema)
    provider = get_model_provider(model_role)
    llm_options = {"temperature": 0}
    if provider == "google_genai":
        llm_options.update(
            timeout=settings.nlq_model_timeout_seconds,
            max_retries=0,
            thinking_config={"thinking_level": "low"},
        )
    llm = get_llm(model_role, **llm_options)
    if provider == "google_genai" and request_interval_seconds is None:
        request_interval_seconds = 4.2  # stay below Gemini Free Tier's 15 RPM ceiling
    evaluation_started = time.perf_counter()
    last_request_started = None
    results = []
    for case in cases:
        result = {"id": case["id"], "question": case["question"]}
        prompt = (
            "Generate one PostgreSQL SELECT query for the user's question. "
            "Return only SQL, no explanation. Use only tables and columns in the schema. "
            "Do not modify data. Return every requested answer dimension and metric, without unrelated columns. "
            "Round currency and average-price outputs to 2 decimal places. For top-N results, add a deterministic "
            "secondary ORDER BY on the displayed name or ID to break ties. For revenue, exclude orders whose "
            "status is 'đã hủy'; calculate SUM(quantity * unit_price), never shipping_fee.\n\nSCHEMA:\n" + schema +
            "\nQUESTION:\n" + case["question"]
        )
        try:
            if request_interval_seconds is not None and last_request_started is not None:
                wait_seconds = request_interval_seconds - (time.perf_counter() - last_request_started)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            generation_started = time.perf_counter()
            last_request_started = generation_started
            if trace_langfuse:
                with langfuse_observation(
                    name="nlq-sql-generation",
                    input={"question": case["question"], "schema": schema},
                ) as observation:
                    handler = get_langfuse_handler()
                    if handler is None:
                        raise RuntimeError("Langfuse callback handler could not be created.")
                    response = _invoke_model_response(llm, prompt, [handler])
                    sql = _extract_sql(extract_message_text(response))
                    update_langfuse_observation(
                        observation,
                        output={"generated_sql": sql},
                    )
            else:
                response = _invoke_model_response(llm, prompt, [])
                sql = _extract_sql(extract_message_text(response))
            result["generation_latency_seconds"] = round(time.perf_counter() - generation_started, 3)
            result.update(_response_usage(response))
            result["generated_sql"] = sql
            try:
                safe_sql = validate_sql(sql, allowed_tables, allowed_columns)
            except (ValueError, ParseError) as exc:
                result["status"] = "rejected"
                result["reason"] = str(exc)
                results.append(result)
                print(json.dumps(result, ensure_ascii=True))
                continue
            _, ordered = _sql_features(safe_sql)
            result["validated_sql"] = safe_sql
            with psycopg.connect(settings.nlq_database_url, connect_timeout=3) as conn:
                conn.execute("SET TRANSACTION READ ONLY")
                conn.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (f"{settings.nlq_statement_timeout_ms}ms",),
                )
                cursor = conn.execute(safe_sql)
                actual = [dict(zip((col.name for col in cursor.description), row))
                          for row in cursor.fetchall()]
            expected_value = case["expected"]
            expected = expected_value if isinstance(expected_value, list) else [expected_value]
            ok = _execution_results_match(actual, expected, ordered=ordered)
            relaxed_ok = _relaxed_execution_results_match(actual, expected, ordered=ordered)
            result.update({
                "status": "pass" if ok else "fail",
                "strict_status": "pass" if ok else "fail",
                "relaxed_status": "pass" if relaxed_ok else "fail",
                "expected": expected,
                "actual": actual,
            })
        except Exception as exc:
            result.update({
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc).replace("\n", " ")[:500],
            })
        results.append(result)
        print(json.dumps(result, ensure_ascii=True, default=str))

    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("pass", "fail", "rejected", "error")
    }
    strict_correct = sum(result.get("strict_status") == "pass" for result in results)
    relaxed_correct = sum(result.get("relaxed_status") == "pass" for result in results)
    total = len(cases)
    generated = [result for result in results if "generation_latency_seconds" in result]
    token_totals = {
        key: sum(result.get(key, 0) for result in generated)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    if trace_langfuse:
        flush_langfuse()
    model_name = MODEL_REGISTRY.get(
        model_role,
        settings.default_model if model_role == "default" else model_role,
    )
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model_name)
    if model_name.startswith("ollama/"):
        estimated_api_cost = 0.0
        pricing_basis = "local inference; excludes electricity and hardware"
    elif pricing:
        estimated_api_cost = (
            token_totals["input_tokens"] * pricing["input"]
            + token_totals["output_tokens"] * pricing["output"]
        ) / 1_000_000
        pricing_basis = "Google Gemini paid-tier standard text rates"
    else:
        estimated_api_cost = None
        pricing_basis = "pricing not configured for this model"

    return {
        "model_role": model_role,
        "model_name": model_name,
        "cases": total,
        "execution_correct": counts["pass"],
        "execution_accuracy": counts["pass"] / total if total else 0,
        "strict_execution_correct": strict_correct,
        "strict_execution_accuracy": strict_correct / total if total else 0,
        "relaxed_execution_correct": relaxed_correct,
        "relaxed_execution_accuracy": relaxed_correct / total if total else 0,
        "executed_cases": counts["pass"] + counts["fail"],
        "executed_accuracy": (
            counts["pass"] / (counts["pass"] + counts["fail"])
            if counts["pass"] + counts["fail"] else 0
        ),
        "counts": counts,
        "mean_generation_latency_seconds": (
            sum(result["generation_latency_seconds"] for result in generated) / len(generated)
            if generated else None
        ),
        "total_generation_latency_seconds": round(
            sum(result["generation_latency_seconds"] for result in generated), 3
        ),
        "evaluation_wall_seconds": round(time.perf_counter() - evaluation_started, 3),
        "token_totals": token_totals,
        "estimated_api_cost_usd": round(estimated_api_cost, 8) if estimated_api_cost is not None else None,
        "pricing_basis": pricing_basis,
        "model_evaluated": True,
        "evaluation_scope": "SQL generation with seed DDL; not the production LangGraph/schema-linking path",
        "trace_uploaded": trace_langfuse,
        "results": results,
    }


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _ensure_nlq_dataset(client, dataset_name: str, result: dict) -> list:
    from langfuse.api.commons.errors.not_found_error import NotFoundError

    try:
        dataset = client.get_dataset(dataset_name, fetch_items_page_size=100)
    except NotFoundError:
        client.create_dataset(
            name=dataset_name,
            description="Synthetic NLQ golden set with expected PostgreSQL result rows.",
            metadata={"version": "v1", "data_classification": "synthetic"},
        )
        dataset = client.get_dataset(dataset_name, fetch_items_page_size=100)

    existing_items = {item.id: item for item in dataset.items}
    expected_by_id = {
        case["id"]: case["expected"]
        if isinstance(case["expected"], list) else [case["expected"]]
        for case in json.loads(Path("data/nlq_golden_set.json").read_text(encoding="utf-8"))
    }
    dataset_ids = set()
    for case_result in result["results"]:
        case_id = case_result["id"]
        if case_id not in expected_by_id:
            raise ValueError(f"NLQ result contains unknown golden case '{case_id}'.")
        item_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"ai-assistant-nlq-v1-{case_id}"
        ))
        dataset_ids.add(item_id)
        expected = {"rows": _json_safe(expected_by_id[case_id])}
        existing = existing_items.get(item_id)
        if existing is not None:
            if (
                existing.input.get("question") != case_result["question"]
                or existing.expected_output != expected
            ):
                raise RuntimeError(
                    f"Langfuse item {case_id} differs from the checked-in golden case; "
                    "use a new dataset version/name instead of reusing stale data."
                )
            continue
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,
            input={"question": case_result["question"]},
            expected_output=expected,
            metadata={"case_id": case_id},
        )

    dataset = client.get_dataset(dataset_name, fetch_items_page_size=100)
    items = [item for item in dataset.items if item.id in dataset_ids]
    if len(items) != len(dataset_ids):
        raise RuntimeError(
            f"Langfuse dataset returned {len(items)} of {len(dataset_ids)} expected NLQ items."
        )
    return items


def _aggregate_nlq_run(*, item_results, **kwargs):
    counts = {status: 0 for status in ("pass", "fail", "rejected", "error")}
    relaxed_passes = 0
    for item_result in item_results:
        status = item_result.output.get("status")
        if status in counts:
            counts[status] += 1
        if item_result.output.get("relaxed_status", status) == "pass":
            relaxed_passes += 1
    total = sum(counts.values())
    return [
        Evaluation(
            name="execution_accuracy",
            value=counts["pass"] / total if total else 0,
            comment=f"Strict: {counts['pass']} matching results out of {total} NLQ cases.",
            data_type="NUMERIC",
        ),
        Evaluation(
            name="execution_accuracy_relaxed",
            value=relaxed_passes / total if total else 0,
            comment=f"Relaxed diagnostic: {relaxed_passes} matching results out of {total} NLQ cases.",
            data_type="NUMERIC",
        ),
        *[
            Evaluation(
                name=f"{status}_cases",
                value=count,
                data_type="NUMERIC",
            )
            for status, count in counts.items()
        ],
    ]


def upload_langfuse_experiment(
    result: dict,
    dataset_name: str = LANGFUSE_DATASET_NAME,
) -> dict:
    """Publish a completed local NLQ evaluation without rerunning the model."""
    if not is_langfuse_configured():
        raise RuntimeError("Langfuse Cloud is not configured; set valid Langfuse keys in .env.")
    if not result.get("results") or result.get("cases") != len(result["results"]):
        raise ValueError("The NLQ result must contain one result for every evaluated case.")

    init_tracing()
    client = get_client()
    dataset_items = _ensure_nlq_dataset(client, dataset_name, result)
    result_by_id = {item["id"]: _json_safe(item) for item in result["results"]}

    def task(*, item, **kwargs):
        metadata = item.metadata if hasattr(item, "metadata") else item["metadata"]
        case_id = metadata["case_id"]
        return result_by_id[case_id]

    def execution_correct(*, output, **kwargs):
        return Evaluation(
            name="execution_correct",
            value=output.get("status") == "pass",
            comment=output.get("reason") or output.get("error_type"),
            data_type="BOOLEAN",
        )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = client.run_experiment(
        name=f"NLQ model evaluation — {result['model_role']}",
        run_name=f"{dataset_name}-{result['model_role']}-{run_stamp}",
        description=(
            "Replay of locally generated SQL and read-only execution results. "
            "The model is not called again by this Langfuse experiment."
        ),
        data=dataset_items,
        task=task,
        evaluators=[execution_correct],
        run_evaluators=[_aggregate_nlq_run],
        max_concurrency=1,
        metadata={
            "model_role": result["model_role"],
            "case_count": result["cases"],
            "evaluation_mode": "precomputed local result replay",
            "data_classification": "synthetic",
            "user_approved_cloud_upload": True,
        },
    )
    client.flush()
    return {
        "dataset": dataset_name,
        "items": len(experiment.item_results),
        "experiment_url": experiment.experiment_url,
        "scores": [
            {"name": score.name, "value": score.value}
            for score in experiment.run_evaluations
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-role", choices=("local", "fast", "strong", "default"), default="local")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=4.2,
        help="Minimum spacing between model calls; keeps Gemini Free Tier runs below 15 RPM.",
    )
    parser.add_argument("--output", type=Path, help="Write the complete result JSON to this path.")
    parser.add_argument("--langfuse", action="store_true",
                        help="Upload this completed synthetic evaluation to Langfuse.")
    parser.add_argument(
        "--trace-langfuse",
        action="store_true",
        help="Send one Langfuse generation trace per newly executed model call.",
    )
    parser.add_argument("--upload-results", type=Path,
                        help="Upload an existing evaluation JSON without rerunning the model.")
    parser.add_argument("--dataset-name", default=LANGFUSE_DATASET_NAME)
    args = parser.parse_args()
    if args.upload_results and not args.langfuse:
        parser.error("--upload-results requires the explicit --langfuse opt-in.")
    if args.trace_langfuse and args.upload_results:
        parser.error("--trace-langfuse requires model execution; it cannot replay a saved file.")
    if args.upload_results:
        result = json.loads(args.upload_results.read_text(encoding="utf-8"))
    else:
        result = evaluate(
            args.model_role,
            args.limit,
            trace_langfuse=args.trace_langfuse,
            request_interval_seconds=args.request_interval_seconds,
        )
    if args.langfuse:
        result["langfuse"] = upload_langfuse_experiment(result, args.dataset_name)
    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"},
                     ensure_ascii=True, indent=2))
