"""Evaluate model-generated SQL against the NLQ golden set, read-only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

from langfuse import Evaluation, get_client
import psycopg
from sqlglot import exp, parse

from app.core.config import settings
from app.core.llm import get_llm
from app.core.tracing import init_tracing, is_langfuse_configured
from scripts.evaluate_nlq_golden import is_read_only_query, results_match

LANGFUSE_DATASET_NAME = "ai-assistant-nlq-golden-v1"


def _sql_features(sql: str) -> tuple[int, bool]:
    """Return parsed statement count and whether the outer query orders rows."""
    statements = [
        statement for statement in parse(sql, read="postgres")
        if statement is not None and not isinstance(statement, exp.Semicolon)
    ]
    has_order_by = len(statements) == 1 and statements[0].args.get("order") is not None
    return len(statements), has_order_by


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
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than zero.")
        cases = cases[:limit]
    schema = _load_schema_ddl()
    llm = get_llm(model_role, temperature=0)
    results = []
    for case in cases:
        result = {"id": case["id"], "question": case["question"]}
        prompt = (
            "Generate one PostgreSQL SELECT query for the user's question. "
            "Return only SQL, no explanation. Use only tables and columns in the schema. "
            "Do not modify data.\n\nSCHEMA:\n" + schema +
            "\nQUESTION:\n" + case["question"]
        )
        try:
            sql = _extract_sql(str(llm.invoke(prompt).content))
            result["generated_sql"] = sql
            statement_count, ordered = _sql_features(sql)
            if (not is_read_only_query(sql)
                    or statement_count != 1):
                result["status"] = "rejected"
                result["reason"] = "non-single-read-only SQL"
                results.append(result)
                print(json.dumps(result, ensure_ascii=True))
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
            result.update({
                "status": "pass" if ok else "fail",
                "expected": expected,
                "actual": actual,
            })
        except Exception as exc:
            result.update({
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        results.append(result)
        print(json.dumps(result, ensure_ascii=True, default=str))

    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("pass", "fail", "rejected", "error")
    }
    total = len(cases)
    return {
        "model_role": model_role,
        "cases": total,
        "execution_correct": counts["pass"],
        "execution_accuracy": counts["pass"] / total if total else 0,
        "executed_cases": counts["pass"] + counts["fail"],
        "executed_accuracy": (
            counts["pass"] / (counts["pass"] + counts["fail"])
            if counts["pass"] + counts["fail"] else 0
        ),
        "counts": counts,
        "model_evaluated": True,
        "trace_uploaded": False,
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
    for item_result in item_results:
        status = item_result.output.get("status")
        if status in counts:
            counts[status] += 1
    total = sum(counts.values())
    return [
        Evaluation(
            name="execution_accuracy",
            value=counts["pass"] / total if total else 0,
            comment=f"{counts['pass']} matching results out of {total} NLQ cases.",
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
    parser.add_argument("--output", type=Path, help="Write the complete result JSON to this path.")
    parser.add_argument("--langfuse", action="store_true",
                        help="Upload this completed synthetic evaluation to Langfuse.")
    parser.add_argument("--upload-results", type=Path,
                        help="Upload an existing evaluation JSON without rerunning the model.")
    parser.add_argument("--dataset-name", default=LANGFUSE_DATASET_NAME)
    args = parser.parse_args()
    if args.upload_results and not args.langfuse:
        parser.error("--upload-results requires the explicit --langfuse opt-in.")
    if args.upload_results:
        result = json.loads(args.upload_results.read_text(encoding="utf-8"))
    else:
        result = evaluate(args.model_role, args.limit)
    if args.langfuse:
        result["langfuse"] = upload_langfuse_experiment(result, args.dataset_name)
    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"},
                     ensure_ascii=True, indent=2))
