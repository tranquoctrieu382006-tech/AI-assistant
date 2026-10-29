"""Analyze retrieval score overlap and fixed cutoff tradeoffs on development only."""
import json
from pathlib import Path

from app.core.vector_store import close_pool, similarity_search


ROOT = Path(__file__).resolve().parents[1]
DEV_PATH = ROOT / "data/rag_golden_set.json"
OUT_PATH = ROOT / "eval/dev_threshold_analysis_m1.json"
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "max": max(values) if values else None,
    }


def analyze(records: list[dict], thresholds: list[float] = THRESHOLDS) -> dict:
    distributions = {}
    for category in ("answerable", "in_scope_missing", "out_of_scope_far"):
        distributions[category] = summarize([
            item["top_score"] for item in records
            if item["group"] == category and item["top_score"] is not None
        ])
    sweeps = []
    answerable_count = distributions["answerable"]["count"]
    for threshold in thresholds:
        row = {
            "threshold": threshold,
            "answerable_cases": answerable_count,
            "answerable_rejected_count": sum(
                item["top_score"] is None or item["top_score"] < threshold
                for item in records if item["group"] == "answerable"
            ),
        }
        row["answerable_rejected_rate"] = (
            row["answerable_rejected_count"] / answerable_count
            if answerable_count else None
        )
        for group in ("in_scope_missing", "out_of_scope_far"):
            group_records = [item for item in records if item["group"] == group]
            false_positives = sum(
                item["top_score"] is not None and item["top_score"] >= threshold
                for item in group_records
            )
            row[group] = {
                "cases": len(group_records),
                "false_positive_count": false_positives,
                "false_positive_rate": (
                    false_positives / len(group_records)
                    if len(group_records) >= 10 else None
                ),
                "rate_note": None if len(group_records) >= 10 else "Fewer than 10 cases; counts only.",
            }
        sweeps.append(row)
    return {
        "dataset": "development M1 only",
        "definition": "Accepted when top retrieval score >= threshold; false positive means accepting an expected-abstention case.",
        "thresholds": thresholds,
        "score_distributions": distributions,
        "tradeoffs": sweeps,
        "limitation": "Descriptive development analysis; thresholds are not validated on held-out data.",
    }


def main() -> int:
    cases = json.loads(DEV_PATH.read_text(encoding="utf-8"))
    records = []
    try:
        for case in cases:
            sources = case.get("expected_sources")
            if sources is None:
                sources = [case["source"]] if case.get("source") else []
            group = "answerable" if sources else case["category"]
            results = similarity_search(case["question"], k=5, corpus="demo")
            records.append({
                "group": group,
                "top_score": results[0]["score"] if results else None,
            })
    finally:
        close_pool()
    report = analyze(records)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {OUT_PATH.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
