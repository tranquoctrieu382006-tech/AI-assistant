"""Run the current structured RAG answer chain locally without uploading traces."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time

from langchain_core.exceptions import OutputParserException
from langfuse import get_client

from app.core.config import settings
from app.core.llm import get_llm
from app.core.rag_prompt import NO_ANSWER, build_rag_chain
from app.core.tracing import init_tracing, is_langfuse_configured
from scripts.evaluate_rag_experiment import _normalize_answer_fragment

logger = logging.getLogger(__name__)


def run_evaluation(
    *,
    model_role: str = "local",
    dataset_name: str = "ai-assistant-rag-golden-v3",
    limit: int = 30,
) -> dict:
    if limit < 1:
        raise ValueError("limit must be greater than zero.")
    if not is_langfuse_configured():
        raise RuntimeError(
            "Langfuse credentials are required to read the existing golden contexts."
        )

    init_tracing()
    dataset = get_client().get_dataset(dataset_name, fetch_items_page_size=100)
    def item_order(item):
        case_id = (item.metadata or {}).get("case_id", item.id)
        match = re.search(r"(\d+)$", case_id)
        return (int(match.group(1)) if match else float("inf"), case_id)

    items = sorted(dataset.items, key=item_order)
    if not items:
        raise RuntimeError(f"Langfuse dataset '{dataset_name}' contains no items.")
    items = items[:limit]

    chain = build_rag_chain(get_llm(model_role, temperature=0))
    records = []
    counts: Counter[str] = Counter()
    for item in items:
        metadata = item.metadata or {}
        item_input = item.input
        context = item_input.get("context", "")
        expected_value = (item.expected_output or {}).get("answer_contains", [])
        expected_parts = expected_value if isinstance(expected_value, list) else [expected_value]
        record = {
            "id": metadata.get("case_id", item.id),
            "category": metadata.get("category", "unknown"),
            "has_context": bool(context),
            "status": "no_context",
            "answer": NO_ANSWER,
            "latency_ms": 0.0,
            "parse_error": None,
            "answer_contains": None,
            "answer_contains_normalized": None,
            "abstention_correct": None,
        }

        if context:
            started = time.perf_counter()
            try:
                generated = chain.invoke({
                    "question": item_input["question"],
                    "context": context,
                })
            except OutputParserException as exc:
                record["status"] = "parse_error"
                record["parse_error"] = type(exc).__name__
                logger.warning("Structured RAG parse failed for %s.", record["id"])
            except Exception as exc:
                record["status"] = "generation_error"
                record["parse_error"] = type(exc).__name__
                logger.warning(
                    "Structured RAG generation failed for %s (%s).",
                    record["id"],
                    type(exc).__name__,
                )
            else:
                record["status"] = "abstained" if generated.abstain else "success"
                record["answer"] = (
                    NO_ANSWER if generated.abstain else generated.answer.strip()
                )
            record["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 2
            )

        counts[record["status"]] += 1
        if expected_parts and record["status"] in {"success", "abstained", "no_context"}:
            answer = record["answer"]
            record["answer_contains"] = all(
                part.casefold() in answer.casefold() for part in expected_parts
            )
            normalized_answer = _normalize_answer_fragment(answer)
            record["answer_contains_normalized"] = all(
                _normalize_answer_fragment(part) in normalized_answer
                for part in expected_parts
            )
        expected_sources = metadata.get("expected_sources", [])
        if record["status"] in {"success", "abstained", "no_context"}:
            record["abstention_correct"] = (
                (not expected_sources) == (record["answer"] == NO_ANSWER)
            )
        records.append(record)
        print(json.dumps({
            "id": record["id"],
            "status": record["status"],
            "latency_ms": record["latency_ms"],
            "answer_contains": record["answer_contains"],
            "answer_contains_normalized": record["answer_contains_normalized"],
        }, ensure_ascii=True))

    counts["total"] = len(records)
    counts["model_calls"] = sum(
        record["status"] in {"success", "abstained", "parse_error", "generation_error"}
        for record in records
    )
    counts["parse_rate_denominator"] = counts["model_calls"]
    counts["parse_failures"] = counts["parse_error"]
    evaluable = [
        record for record in records
        if record["answer_contains_normalized"] is not None
    ]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_role": model_role,
        "model": settings.local_model if model_role == "local" else model_role,
        "prompt_schema": "RagGeneratedAnswer(answer, abstain)",
        "dataset": dataset_name,
        "dataset_items": len(dataset.items),
        "evaluated_cases": len(records),
        "counts": dict(counts),
        "parse_failure_rate": (
            counts["parse_failures"] / counts["parse_rate_denominator"]
            if counts["parse_rate_denominator"] else None
        ),
        "normalized_answer_match_count": sum(
            record["answer_contains_normalized"] is True for record in evaluable
        ),
        "normalized_answer_match_denominator": len(evaluable),
        "records": records,
        "data_handling": (
            "Reads synthetic question/context from the existing Langfuse dataset. "
            "Model calls run locally without Langfuse callbacks; results are written "
            "only to the local output file."
        ),
    }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-role", default="local")
    parser.add_argument("--dataset-name", default="ai-assistant-rag-golden-v3")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/rag_structured_local_2026-10-23.json"),
    )
    args = parser.parse_args()
    result = run_evaluation(
        model_role=args.model_role,
        dataset_name=args.dataset_name,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "model": result["model"],
        "evaluated_cases": result["evaluated_cases"],
        "counts": result["counts"],
        "parse_failure_rate": result["parse_failure_rate"],
        "normalized_answer_matches": (
            f"{result['normalized_answer_match_count']}/"
            f"{result['normalized_answer_match_denominator']}"
        ),
    }, ensure_ascii=True, indent=2))
