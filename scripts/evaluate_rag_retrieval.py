"""Report retrieval Hit@k, MRR, and per-category metrics."""
import json
from pathlib import Path, PurePosixPath

from app.core.config import settings
from scripts.rag_eval_metrics import context_recall_at_k


def source_matches(actual: str, expected: str) -> bool:
    """Compare corpus-relative source paths, not potentially ambiguous basenames."""
    actual_path = PurePosixPath(actual.replace("\\", "/"))
    expected_path = PurePosixPath(expected.replace("\\", "/"))
    return actual_path == expected_path


def _source_targets(case: dict) -> list[str]:
    sources = case.get("expected_sources")
    if sources is not None:
        return sources
    return [case["source"]] if case.get("source") else []


def _retrieval_metrics(records: list[dict], k: int) -> dict:
    answerable = [record for record in records if record["expected_sources"]]
    if not answerable:
        return {
            f"hit@{cutoff}": None for cutoff in (1, 3, 5)
        } | {
            "mrr": None,
            f"expected_source_coverage@{k}": None,
            "answerable_cases": 0,
        }

    output = {"answerable_cases": len(answerable)}
    for cutoff in (1, 3, 5):
        output[f"hit@{cutoff}"] = sum(
            all(any(source_matches(item["source"], expected)
                    for item in record["results"][:cutoff])
                for expected in record["expected_sources"])
            for record in answerable
        ) / len(answerable)
    output["mrr"] = sum(
        1 / min(record["ranks"]) if record["ranks"] else 0
        for record in answerable
    ) / len(answerable)
    output[f"expected_source_coverage@{k}"] = sum(
        sum(any(source_matches(item["source"], expected)
                for item in record["results"][:k])
            for expected in record["expected_sources"]) / len(record["expected_sources"])
        for record in answerable
    ) / len(answerable)
    return output


def _category_metrics(records: list[dict], abstentions: list[dict], k: int) -> dict:
    categories = sorted(
        {record["category"] for record in records}
        | {record["category"] for record in abstentions}
    )
    output = {}
    for category in categories:
        category_abstentions = [
            record for record in abstentions if record["category"] == category
        ]
        metrics = _retrieval_metrics(
            [record for record in records if record["category"] == category], k
        )
        metrics["abstention_cases"] = len(category_abstentions)
        metrics["abstention_accuracy"] = (
            sum(record["correct"] for record in category_abstentions) / len(category_abstentions)
            if category_abstentions else None
        )
        output[category] = metrics
    return output


def _abstention_risk_summary(abstentions: list[dict]) -> dict:
    output = {}
    for category in sorted({record["category"] for record in abstentions}):
        group = [record for record in abstentions if record["category"] == category]
        false_positives = sum(not record["correct"] for record in group)
        output[category] = {
            "cases": len(group),
            "false_positive_count": false_positives,
            "false_positive_rate": false_positives / len(group) if len(group) >= 10 else None,
            "rate_reporting_note": None if len(group) >= 10 else "Fewer than 10 cases; report counts only.",
        }
    return output


def evaluate(k: int = 5) -> dict:
    from app.core.vector_store import similarity_search
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    records = []
    abstentions = []
    failure_examples = []
    context_recall_scores = []
    false_negative_counts = {}
    abstention_correct = 0
    for case in cases:
        results = similarity_search(case["question"], k=max(k, 5), corpus="demo")
        expected_sources = _source_targets(case)
        if expected_sources:
            context_recall = context_recall_at_k(case, results, k)
            if context_recall is not None:
                context_recall_scores.append(context_recall)
            predicted_abstain = (
                not results or results[0]["score"] < settings.rag_min_relevance_score
            )
            if predicted_abstain:
                false_negative_counts[case.get("category", "simple")] = (
                    false_negative_counts.get(case.get("category", "simple"), 0) + 1
                )
            ranks = [
                next((i for i, item in enumerate(results, 1)
                      if source_matches(item["source"], expected)), None)
                for expected in expected_sources
            ]
            records.append({
                "category": case.get("category", "simple"),
                "expected_sources": expected_sources,
                "ranks": [rank for rank in ranks if rank is not None],
                "results": results,
            })
            if not all(ranks):
                found = [
                    {
                        "source": item["source"],
                        "rank": index,
                        "score": item["score"],
                    }
                    for index, item in enumerate(results[:k], 1)
                ]
                failure_examples.append({
                    "id": case["id"],
                    "category": case.get("category", "simple"),
                    "question": case["question"],
                    "expected_sources": expected_sources,
                    "expected_source_ranks": ranks,
                    "retrieved_top_sources": found,
                })
            print(f"{case['id']}: {'HIT' if all(ranks) else 'MISS'} "
                  f"ranks={ranks} top={results[0]['source'] if results else '-'}")
        else:
            abstained = not results or results[0]["score"] < settings.rag_min_relevance_score
            abstention_correct += int(abstained)
            abstentions.append({
                "category": case.get("category", "out_of_scope"),
                "correct": abstained,
            })
            if not abstained:
                failure_examples.append({
                    "id": case["id"],
                    "category": case.get("category", "out_of_scope"),
                    "question": case["question"],
                    "expected_sources": [],
                    "expected_abstention": True,
                    "retrieved_top_sources": [
                        {
                            "source": item["source"],
                            "rank": index,
                            "score": item["score"],
                        }
                        for index, item in enumerate(results[:k], 1)
                    ],
                })
            print(f"{case['id']}: {'ABSTAIN PASS' if abstained else 'ABSTAIN FAIL'} "
                  f"top_score={results[0]['score'] if results else '-'}")

    answerable = _retrieval_metrics(records, k)
    categories = _category_metrics(records, abstentions, k)
    total_abstentions = len(abstentions)
    abstention_risk = _abstention_risk_summary(abstentions)
    answerable_count = len(records)
    return {
        "answerable_cases": answerable["answerable_cases"],
        "hit@1": answerable["hit@1"],
        "hit@3": answerable["hit@3"],
        "hit@5": answerable["hit@5"],
        "mrr": answerable["mrr"],
        f"expected_source_coverage@{k}": answerable[f"expected_source_coverage@{k}"],
        f"context_recall@{k}": (
            sum(context_recall_scores) / len(context_recall_scores)
            if context_recall_scores else None
        ),
        "evidence_cases": len(context_recall_scores),
        "abstention_risk_by_type": abstention_risk,
        "false_negative_count_answerable": sum(false_negative_counts.values()),
        "false_negative_rate_answerable": (
            sum(false_negative_counts.values()) / answerable_count
            if answerable_count >= 10 else None
        ),
        "false_positive_definition": "System returned retrieval results above the cutoff for a case expected to abstain.",
        "by_category": categories,
        "failure_examples": failure_examples,
        "abstention_cases": total_abstentions,
        "abstention_accuracy": abstention_correct / total_abstentions if total_abstentions else None,
        "abstention_note": "Small sample; report count and do not treat the rate as a stable quality estimate.",
    }


if __name__ == "__main__":
    from app.core.vector_store import close_pool
    try:
        print(json.dumps(evaluate(settings.rag_top_k), ensure_ascii=False, indent=2))
    finally:
        close_pool()
