"""Evaluate and compare 5 retrieval and reranking configurations on the golden set."""
from __future__ import annotations

import json
import time
from pathlib import Path, PurePosixPath

from app.core.config import settings
from app.core.retriever import retrieve_and_gate, clear_bm25_cache


def source_matches(actual: str, expected: str) -> bool:
    actual_path = PurePosixPath(actual.replace("\\", "/"))
    expected_path = PurePosixPath(expected.replace("\\", "/"))
    return actual_path == expected_path


def _source_targets(case: dict) -> list[str]:
    sources = case.get("expected_sources")
    if sources is not None:
        return sources
    return [case["source"]] if case.get("source") else []


def evaluate_config(
    name: str,
    mode: str,
    k: int,
    use_reranker: bool = False,
    reranker_top_n: int = 5,
) -> dict:
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    answerable_records = []
    abstention_records = []
    false_negative_count = 0
    ranks_by_case = {}
    latencies = []

    for case in cases:
        expected_sources = _source_targets(case)

        t0 = time.perf_counter()
        matches, is_refused = retrieve_and_gate(
            case["question"],
            top_k=k,
            mode=mode,
            use_reranker=use_reranker,
            reranker_top_n=reranker_top_n,
        )
        t_elapsed = time.perf_counter() - t0
        latencies.append(t_elapsed)

        effective_matches = matches[:reranker_top_n if use_reranker else k]

        if expected_sources:
            if is_refused:
                false_negative_count += 1

            ranks = [
                next((i for i, item in enumerate(effective_matches, 1)
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
                "matches": effective_matches,
            })
        else:
            correct = is_refused
            abstention_records.append({
                "id": case["id"],
                "category": case.get("category", "out_of_scope"),
                "correct": correct,
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

    in_scope_missing = [r for r in abstention_records if r["category"] == "in_scope_missing"]
    out_of_scope_far = [r for r in abstention_records if r["category"] == "out_of_scope_far"]

    return {
        "name": name,
        "mode": mode,
        "k_candidates": k,
        "use_reranker": use_reranker,
        "top_n": reranker_top_n if use_reranker else k,
        "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
        "hit@1": hit_1,
        "hit@3": hit_3,
        "hit@5": hit_5,
        "mrr": mrr,
        "expected_source_coverage@5": coverage_5,
        "false_negative_answerable": false_negative_count,
        "in_scope_missing_fp": sum(not r["correct"] for r in in_scope_missing),
        "in_scope_missing_total": len(in_scope_missing),
        "out_of_scope_far_fp": sum(not r["correct"] for r in out_of_scope_far),
        "out_of_scope_far_total": len(out_of_scope_far),
    }


def main():
    clear_bm25_cache()
    configs = [
        {"name": "V5", "mode": "vector", "k": 5, "use_reranker": False, "reranker_top_n": 5},
        {"name": "V10", "mode": "vector", "k": 10, "use_reranker": False, "reranker_top_n": 10},
        {"name": "H5", "mode": "hybrid", "k": 5, "use_reranker": False, "reranker_top_n": 5},
        {"name": "R10", "mode": "vector", "k": 10, "use_reranker": True, "reranker_top_n": 5},
        {"name": "HR10", "mode": "hybrid", "k": 10, "use_reranker": True, "reranker_top_n": 5},
    ]

    results = []
    for cfg in configs:
        print(f"Running config {cfg['name']} (mode={cfg['mode']}, k={cfg['k']}, rerank={cfg['use_reranker']})...")
        res = evaluate_config(**cfg)
        results.append(res)
        print(f"  Hit@1={res['hit@1']:.4f}, Hit@3={res['hit@3']:.4f}, MRR={res['mrr']:.4f}, Latency={res['avg_latency_ms']:.1f}ms")

    out_json = Path("eval/rerank_config_comparison.json")
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate Markdown table
    lines = [
        "# So sánh thực nghiệm 5 cấu hình Retrieval & Reranker",
        "",
        "> **Dataset:** `data/rag_golden_set.json` (65 cases)",
        "> **Model Reranker:** `BAAI/bge-reranker-v2-m3`",
        "",
        "| Cấu hình | Retrieval | k lấy | Rerank | Top n | Hit@1 | Hit@3 | Hit@5 | MRR | FP (in-missing) | FP (out-far) | FN (ans) | Độ trễ TB |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| **{r['name']}** | {r['mode']} | {r['k_candidates']} | {'bge-reranker' if r['use_reranker'] else 'Không'} | {r['top_n']} | "
            f"{r['hit@1']:.4f} | {r['hit@3']:.4f} | {r['hit@5']:.4f} | {r['mrr']:.4f} | "
            f"{r['in_scope_missing_fp']}/{r['in_scope_missing_total']} | {r['out_of_scope_far_fp']}/{r['out_of_scope_far_total']} | "
            f"{r['false_negative_answerable']}/41 | {r['avg_latency_ms']:.1f} ms |"
        )

    lines.extend([
        "",
        "## Tiêu chuẩn chọn cấu hình mặc định",
        "1. **Hit@1 và MRR không giảm** so với baseline V5.",
        "2. **False positive ở nhóm câu ngoài phạm vi (out_of_scope_far)** không tăng.",
        "3. **Độ trễ trung bình < 3.000 ms** (3 giây) trên môi trường local.",
        "4. *Lưu ý:* Đây là kết quả thăm dò trên development set để tìm kiếm cấu hình tối ưu.",
    ])

    out_md = Path("eval/rerank_config_comparison.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved results to {out_json} and {out_md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
