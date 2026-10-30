"""Aggregate-only RAG held-out evaluator; never emits per-case results.

Frozen-input verification hashes text with LF line endings and binary inputs
verbatim (see ``scripts.manifest_hashing``) so a manifest stays valid across
Windows CRLF and POSIX LF checkouts. Manifests frozen with raw-byte hashing are
no longer verifiable here; regenerate them with
``python -m scripts.write_development_manifest`` /
``python -m scripts.write_heldout_manifest`` before evaluating.
"""
import json
import random
from collections import Counter
from pathlib import Path

from app.core.config import settings
from scripts.evaluate_rag_retrieval import _source_targets
from scripts.manifest_hashing import sha256_file
from scripts.rag_eval_metrics import context_recall_at_k
from scripts.validate_golden_data import validate_rag_cases


ROOT = Path(__file__).resolve().parents[1]
HELDOUT_PATH = ROOT / "eval/rag_heldout_m1.json"
HELDOUT_MANIFEST_PATH = ROOT / "eval/heldout_manifest_m2.json"
M1_MANIFEST_PATH = ROOT / "eval/development_manifest_m1.json"
RESULTS_PATH = ROOT / "eval/rag_heldout_m2_results.json"


def verify_frozen_inputs() -> tuple[dict, list[dict]]:
    m1 = json.loads(M1_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(HELDOUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    if sha256_file(M1_MANIFEST_PATH) != manifest["m1_manifest_sha256"]:
        raise ValueError("M1 manifest hash mismatch")
    if manifest["m1_commit"] != "4c663793007d46b30ed7a7854e543110af437aff":
        raise ValueError("Unexpected M1 commit in held-out manifest")
    if sha256_file(HELDOUT_PATH) != manifest["heldout_sha256"]:
        raise ValueError("Held-out data hash mismatch")
    for relative, expected in m1["corpus_files"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"Frozen corpus hash mismatch: {relative}")
    for relative, info in m1["development_sets"].items():
        if sha256_file(ROOT / relative) != info["sha256"]:
            raise ValueError(f"Development set hash mismatch: {relative}")
    for relative, expected in m1["evaluation_code"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"M1 evaluator hash mismatch: {relative}")
    for relative, expected in manifest["heldout_evaluator_code"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"Held-out evaluator hash mismatch: {relative}")
    errors = validate_rag_cases(cases)
    if errors:
        raise ValueError("Held-out data validation failed")
    expected_counts = manifest["category_counts"]
    actual_counts = dict(Counter(case["category"] for case in cases))
    if actual_counts != expected_counts:
        raise ValueError("Held-out category counts mismatch")
    return manifest, cases


def _bootstrap_mean_interval(
    values: list[float], *, samples: int = 10_000, confidence: float = 0.95, seed: int = 20261023,
) -> dict:
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    return {
        "confidence": confidence,
        "lower": means[int(tail * (samples - 1))],
        "upper": means[int((1 - tail) * (samples - 1))],
        "method": "percentile bootstrap over case-level context recall",
        "samples": samples,
        "seed": seed,
    }


def score_cases(cases: list[dict], retriever, *, k: int = 5, cutoff: float = 0.5) -> dict:
    """Score only aggregate primary/risk metrics; do not return case details."""
    context_scores = []
    negative_counts = Counter()
    false_positive_counts = Counter()
    answerable_cases = 0
    false_negative_count = 0

    for case in cases:
        results = retriever(case["question"], k)
        sources = _source_targets(case)
        answered = bool(results) and results[0]["score"] >= cutoff
        if sources:
            answerable_cases += 1
            score = context_recall_at_k(case, results, k)
            if score is not None:
                context_scores.append(score)
            false_negative_count += int(not answered)
        else:
            category = case["category"]
            negative_counts[category] += 1
            false_positive_counts[category] += int(answered)

    if not context_scores:
        raise ValueError("No answerable evidence cases in held-out data")
    context_mean = sum(context_scores) / len(context_scores)
    risk = {}
    for category, count in sorted(negative_counts.items()):
        false_positives = false_positive_counts[category]
        risk[category] = {
            "cases": count,
            "false_positive_count": false_positives,
            "false_positive_rate": false_positives / count if count >= 10 else None,
            "rate_note": None if count >= 10 else "Fewer than 10 cases; counts only.",
        }
    return {
        "answerable_cases": answerable_cases,
        "primary_metric": {
            "name": f"context_recall@{k}",
            "estimate": context_mean,
            "evidence_cases": len(context_scores),
            "case_level_bootstrap_interval": _bootstrap_mean_interval(context_scores),
        },
        "risk_metric": {
            "definition": "False positive means retrieval returns a result at or above the cutoff for a case expected to abstain.",
            "by_abstention_type": risk,
            "answerable_false_negative_count": false_negative_count,
            "answerable_false_negative_rate": (
                false_negative_count / answerable_cases if answerable_cases >= 10 else None
            ),
        },
        "configuration": {"top_k": k, "relevance_cutoff": cutoff},
        "per_case_output": "omitted",
    }


def main() -> int:
    close_pool = None
    try:
        manifest, cases = verify_frozen_inputs()
        from app.core.vector_store import close_pool, similarity_search

        result = score_cases(
            cases,
            lambda question, k: similarity_search(question, k=k, corpus="demo"),
            k=manifest["top_k"],
            cutoff=manifest["relevance_cutoff"],
        )
        result["heldout_manifest_sha256"] = sha256_file(HELDOUT_MANIFEST_PATH)
        result["m1_commit"] = manifest["m1_commit"]
        RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved aggregate report: {RESULTS_PATH.relative_to(ROOT).as_posix()}")
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": "Held-out evaluation failed; no per-case details were emitted.",
        }, ensure_ascii=False))
        return 1
    finally:
        if close_pool is not None:
            close_pool()


if __name__ == "__main__":
    raise SystemExit(main())
