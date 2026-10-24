"""Evaluate the current production retrieval path on development data only.

Unlike the frozen M1/M2 baseline scripts, this uses retrieve_and_gate with the
same mode, top-k, reranker setting, candidate pool, and cosine gate as the API.
It must not be used to claim held-out quality.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from app.core.config import settings
from app.core.retriever import retrieve_and_gate
from scripts.evaluate_rag_retrieval import _source_targets
from scripts.rag_eval_metrics import context_recall_at_k


def _matches_source(actual: str, expected: str) -> bool:
    return PurePosixPath(actual.replace("\\", "/")) == PurePosixPath(
        expected.replace("\\", "/")
    )


def evaluate_cases(cases: list[dict], retriever=retrieve_and_gate) -> dict:
    k = settings.rag_top_k
    answerable = []
    abstention_counts: dict[str, dict[str, int]] = {}
    recall_scores = []
    answerable_false_negatives = 0
    for case in cases:
        matches, refused = retriever(
            case["question"],
            top_k=k,
            mode=settings.search_mode,
            use_reranker=settings.use_reranker,
            reranker_top_n=k,
            reranker_pool_size=max(k, 20),
        )
        expected_sources = _source_targets(case)
        if expected_sources:
            answerable_false_negatives += int(refused or not matches)
            ranks = [
                next((rank for rank, item in enumerate(matches, 1)
                      if _matches_source(item["source"], source)), None)
                for source in expected_sources
            ]
            evidence_recall = context_recall_at_k(case, matches, k)
            if evidence_recall is not None:
                recall_scores.append(evidence_recall)
            answerable.append({"ranks": ranks, "matches": matches, "expected": expected_sources})
        else:
            category = case.get("category", "out_of_scope")
            counts = abstention_counts.setdefault(category, {"cases": 0, "false_positives": 0})
            counts["cases"] += 1
            # False positive: retrieval passed the gate for a case labeled to abstain.
            counts["false_positives"] += int(not refused and bool(matches))

    metrics = {}
    for cutoff in (1, 3, 5):
        hits = [
            all(any(_matches_source(item["source"], source)
                    for item in record["matches"][:cutoff])
                for source in record["expected"])
            for record in answerable
        ]
        metrics[f"source_hit@{cutoff}"] = sum(hits) / len(hits) if hits else None
    reciprocal_ranks = []
    for row in answerable:
        first_rank = min((rank for rank in row["ranks"] if rank is not None), default=None)
        reciprocal_ranks.append(1 / first_rank if first_rank else 0)
    metrics["first_relevant_source_mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None
    )
    metrics[f"context_recall@{k}"] = (
        sum(recall_scores) / len(recall_scores) if recall_scores else None
    )
    return {
        "dataset": "development",
        "cases": len(cases),
        "answerable_cases": len(answerable),
        "metrics": metrics,
        "abstention_risk": abstention_counts,
        "answerable_false_negative_count": answerable_false_negatives,
        "answerable_false_negative_rate": (
            answerable_false_negatives / len(answerable) if len(answerable) >= 10 else None
        ),
        "configuration": {
            "search_mode": settings.search_mode,
            "top_k": k,
            "cosine_cutoff": settings.rag_min_relevance_score,
            "use_reranker": settings.use_reranker,
            "reranker_candidate_pool": max(k, 20) if settings.use_reranker else k,
            "reranker_top_n": k if settings.use_reranker else None,
            "embedding_model": settings.embedding_model,
        },
        "metric_notes": {
            "source_hit": "All expected source files must appear by the cutoff; this is not evidence-level recall.",
            "first_relevant_source_mrr": "Reciprocal rank of the first expected source; partial multi-source retrieval can score well.",
            "corpus_size_warning": "Current demo corpus is small; top-k metrics are diagnostic only.",
            "heldout_warning": "Development-only evaluator; do not tune from or report as held-out evidence.",
        },
    }


def evaluate() -> dict:
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    return evaluate_cases(cases)


if __name__ == "__main__":
    from app.core.vector_store import close_pool

    try:
        report = evaluate()
        output = Path("eval/production_path_development_latest.json")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved: {output}")
    finally:
        close_pool()
