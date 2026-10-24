"""Evaluate and compare Vector-only vs. Hybrid (BM25 + PGVector) retrieval on golden set."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from app.core.config import settings
from app.core.retriever import retrieve_and_gate, clear_bm25_cache
from scripts.rag_eval_metrics import context_recall_at_k


def source_matches(actual: str, expected: str) -> bool:
    actual_path = PurePosixPath(actual.replace("\\", "/"))
    expected_path = PurePosixPath(expected.replace("\\", "/"))
    return actual_path == expected_path


def _source_targets(case: dict) -> list[str]:
    sources = case.get("expected_sources")
    if sources is not None:
        return sources
    return [case["source"]] if case.get("source") else []


def evaluate_mode(mode: str, k: int = 5) -> dict:
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    answerable_records = []
    abstention_records = []
    false_negative_count = 0
    ranks_by_case = {}

    for case in cases:
        expected_sources = _source_targets(case)
        matches, is_refused = retrieve_and_gate(case["question"], top_k=k, mode=mode)

        if expected_sources:
            # Answerable case
            if is_refused:
                false_negative_count += 1

            ranks = [
                next((i for i, item in enumerate(matches, 1)
                      if source_matches(item["source"], expected)), None)
                for expected in expected_sources
            ]
            ranks_by_case[case["id"]] = ranks
            answerable_records.append({
                "id": case["id"],
                "category": case.get("category", "simple"),
                "expected_sources": expected_sources,
                "ranks": [r for r in ranks if r is not None],
                "all_ranks": ranks,
                "matches": matches,
            })
        else:
            # Expected abstention case
            # System correctly abstains if is_refused is True
            correct = is_refused
            abstention_records.append({
                "id": case["id"],
                "category": case.get("category", "out_of_scope"),
                "correct": correct,
                "top_score": matches[0]["score"] if matches else None,
            })

    total_answerable = len(answerable_records)
    hit_1 = sum(
        all(any(source_matches(item["source"], exp) for item in rec["matches"][:1])
            for exp in rec["expected_sources"])
        for rec in answerable_records
    ) / total_answerable

    hit_3 = sum(
        all(any(source_matches(item["source"], exp) for item in rec["matches"][:3])
            for exp in rec["expected_sources"])
        for rec in answerable_records
    ) / total_answerable

    hit_5 = sum(
        all(any(source_matches(item["source"], exp) for item in rec["matches"][:5])
            for exp in rec["expected_sources"])
        for rec in answerable_records
    ) / total_answerable

    mrr = sum(
        1 / min(rec["ranks"]) if rec["ranks"] else 0
        for rec in answerable_records
    ) / total_answerable

    coverage_5 = sum(
        sum(any(source_matches(item["source"], exp) for item in rec["matches"][:5])
            for exp in rec["expected_sources"]) / len(rec["expected_sources"])
        for rec in answerable_records
    ) / total_answerable

    # Abstention breakdown
    in_scope_missing = [r for r in abstention_records if r["category"] == "in_scope_missing"]
    out_of_scope_far = [r for r in abstention_records if r["category"] == "out_of_scope_far"]

    fp_in_missing = sum(not r["correct"] for r in in_scope_missing)
    fp_out_far = sum(not r["correct"] for r in out_of_scope_far)

    return {
        "mode": mode,
        "k": k,
        "answerable_cases": total_answerable,
        "hit@1": hit_1,
        "hit@3": hit_3,
        "hit@5": hit_5,
        "mrr": mrr,
        "expected_source_coverage@5": coverage_5,
        "false_negative_answerable": false_negative_count,
        "in_scope_missing_total": len(in_scope_missing),
        "in_scope_missing_fp": fp_in_missing,
        "out_of_scope_far_total": len(out_of_scope_far),
        "out_of_scope_far_fp": fp_out_far,
        "ranks_by_case": ranks_by_case,
    }


def main():
    clear_bm25_cache()
    print("Evaluating Vector-only (V5)...")
    v5 = evaluate_mode("vector", k=5)

    print("Evaluating Hybrid BM25 + PGVector (H5)...")
    h5 = evaluate_mode("hybrid", k=5)

    comparison = {
        "vector_only_v5": v5,
        "hybrid_h5": h5,
    }

    out_json = Path("eval/hybrid_vs_vector_comparison.json")
    out_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    md_content = f"""# So sánh Retrieval: Vector-only vs. Hybrid (BM25 + PGVector)

> **Dataset:** `data/rag_golden_set.json` (65 cases: 41 answerable, 19 in_scope_missing, 5 out_of_scope_far)
> **Model Embedding:** `bge-m3`
> **Gating Strategy:** Cosine threshold (`score >= {settings.rag_min_relevance_score}`) from vector branch
> **Hybrid Fusion:** Reciprocal Rank Fusion (EnsembleRetriever with weights [0.5, 0.5], id_key="chunk_id", truncated to top-k)

## Bảng chỉ số tổng hợp

| Chỉ số | Vector-only (V5) | Hybrid RRF (H5) | Chênh lệch (H5 - V5) |
|---|:---:|:---:|:---:|
| **Hit@1** | {v5['hit@1']:.4f} ({int(round(v5['hit@1']*v5['answerable_cases']))}/{v5['answerable_cases']}) | {h5['hit@1']:.4f} ({int(round(h5['hit@1']*h5['answerable_cases']))}/{h5['answerable_cases']}) | {h5['hit@1'] - v5['hit@1']:+.4f} |
| **Hit@3** | {v5['hit@3']:.4f} ({int(round(v5['hit@3']*v5['answerable_cases']))}/{v5['answerable_cases']}) | {h5['hit@3']:.4f} ({int(round(h5['hit@3']*h5['answerable_cases']))}/{h5['answerable_cases']}) | {h5['hit@3'] - v5['hit@3']:+.4f} |
| **Hit@5** | {v5['hit@5']:.4f} ({int(round(v5['hit@5']*v5['answerable_cases']))}/{v5['answerable_cases']}) | {h5['hit@5']:.4f} ({int(round(h5['hit@5']*h5['answerable_cases']))}/{h5['answerable_cases']}) | {h5['hit@5'] - v5['hit@5']:+.4f} |
| **MRR** | {v5['mrr']:.4f} | {h5['mrr']:.4f} | {h5['mrr'] - v5['mrr']:+.4f} |
| **Source Coverage@5** | {v5['expected_source_coverage@5']:.4f} | {h5['expected_source_coverage@5']:.4f} | {h5['expected_source_coverage@5'] - v5['expected_source_coverage@5']:+.4f} |
| **False Negatives (Answerable bị từ chối)** | {v5['false_negative_answerable']}/{v5['answerable_cases']} | {h5['false_negative_answerable']}/{h5['answerable_cases']} | {h5['false_negative_answerable'] - v5['false_negative_answerable']:+d} |
| **False Positives (in_scope_missing)** | {v5['in_scope_missing_fp']}/{v5['in_scope_missing_total']} | {h5['in_scope_missing_fp']}/{h5['in_scope_missing_total']} | {h5['in_scope_missing_fp'] - v5['in_scope_missing_fp']:+d} |
| **False Positives (out_of_scope_far)** | {v5['out_of_scope_far_fp']}/{v5['out_of_scope_far_total']} | {h5['out_of_scope_far_fp']}/{h5['out_of_scope_far_total']} | {h5['out_of_scope_far_fp'] - v5['out_of_scope_far_fp']:+d} |

## Nhận xét & Phân tích
- **Cổng từ chối (Gating):** Do hybrid dùng chung cổng cosine từ nhánh vector, số lượng False Positives của cả 2 nhóm `in_scope_missing` ({v5['in_scope_missing_fp']}/19) và `out_of_scope_far` ({v5['out_of_scope_far_fp']}/5) là đồng nhất.
- **Thứ hạng tài liệu:** BM25 bổ trợ các câu chứa từ khóa chính xác, trong khi vector bge-m3 đảm bảo ngữ nghĩa.
"""
    out_md = Path("eval/hybrid_vs_vector_comparison.md")
    out_md.write_text(md_content, encoding="utf-8")
    print(f"Saved JSON to {out_json} and Markdown to {out_md}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
