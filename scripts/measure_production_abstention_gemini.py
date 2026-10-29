"""
End-to-end production measurement of RAG abstention using Google Gemini 3.5 Flash-Lite.
Measures both the Retrieval Gate (Layer 1) and LLM Grounding Defense (Layer 2)
across all 60 development cases (19 in_scope_missing + 41 answerable).
Strict 4.2s pacing to stay safely under 15 RPM.
"""

import json
from pathlib import Path
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import settings
from app.core.llm import get_llm
from app.core.rag_prompt import (
    NO_ANSWER,
    build_rag_chain,
    format_context,
)
from app.core.vector_store import close_pool, similarity_search

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data/rag_golden_set.json"
OUTPUT_PATH = ROOT / "eval/dev_abstention_prompt_gemini35_m1.json"


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def is_semantic_refusal(text: str) -> bool:
    norm = normalize(text)
    refusal_keywords = [
        "không có thông tin",
        "không chứa đủ thông tin",
        "không được đề cập",
        "không nêu",
        "không có dữ liệu",
        "tài liệu không",
        "không tìm thấy",
    ]
    return any(kw in norm for kw in refusal_keywords)


def main():
    print(f"Loading golden dataset from {GOLDEN_PATH}...")
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"Total cases: {len(cases)}")

    # Nhóm ca kiểm thử: 19 missing-info + 41 answerable
    missing_cases = [c for c in cases if c.get("category") == "in_scope_missing"]
    answerable_cases = [c for c in cases if c.get("category") != "in_scope_missing" and c.get("evidence")]
    print(f"  - Missing info cases: {len(missing_cases)}")
    print(f"  - Answerable cases  : {len(answerable_cases)}")

    llm = get_llm("fast", temperature=0)
    chain = build_rag_chain(llm)

    records = []
    total_calls = 0

    try:
        # Chạy toàn bộ 60 cases
        for idx, case in enumerate(cases, 1):
            cid = case["id"]
            cat = case.get("category", "simple")
            is_missing = (cat == "in_scope_missing")
            expected_abstain = is_missing

            # 1. Retrieval Gate (Layer 1)
            retrieved = similarity_search(case["question"], k=settings.rag_top_k, corpus="demo")
            context_matches = [
                m for m in retrieved
                if m["score"] >= settings.rag_min_relevance_score
            ]
            top_score = retrieved[0]["score"] if retrieved else None
            retrieval_passed = len(context_matches) > 0

            # 2. Generation Layer (Layer 2)
            if not retrieval_passed:
                answer = NO_ANSWER
                model_abstain = True
                defense_layer = "retrieval_gate"
                latency_ms = 0.0
            else:
                context = format_context(context_matches)
                started = time.perf_counter()
                time.sleep(4.2)  # Pacing 4.2s to stay under 15 RPM
                total_calls += 1

                try:
                    result = chain.invoke({"question": case["question"], "context": context})
                    raw_answer = result.answer.strip()
                    model_abstain = result.abstain
                    latency_ms = round((time.perf_counter() - started) * 1000, 1)

                    # Semantic check if model answered with refusal words but set abstain=False
                    if is_semantic_refusal(raw_answer):
                        model_abstain = True

                    answer = raw_answer
                    defense_layer = "llm_generation" if model_abstain else "answered"
                except Exception as exc:
                    answer = f"ERROR: {type(exc).__name__}: {exc}"
                    model_abstain = None
                    defense_layer = "error"
                    latency_ms = round((time.perf_counter() - started) * 1000, 1)

            final_abstain = bool(model_abstain)
            abstention_correct = (final_abstain == expected_abstain)

            parts = case.get("answer_contains", [])
            if isinstance(parts, str):
                parts = [parts]
            fragment_match = (
                all(normalize(part) in normalize(answer) for part in parts)
                if not is_missing and not final_abstain and answer != NO_ANSWER else None
            )

            record = {
                "id": cid,
                "category": cat,
                "question": case["question"],
                "expected_abstain": expected_abstain,
                "top_score": top_score,
                "retrieval_passed": retrieval_passed,
                "context_chunks": len(context_matches),
                "answer": answer,
                "model_abstain": model_abstain,
                "final_abstain": final_abstain,
                "abstention_correct": abstention_correct,
                "fragment_match": fragment_match,
                "defense_layer": defense_layer,
                "latency_ms": latency_ms,
            }
            records.append(record)

            status_icon = "✅" if abstention_correct else "❌"
            print(f"[{idx:02d}/60] {status_icon} {cid} ({cat:<17}): "
                  f"Gate={'PASS' if retrieval_passed else 'BLOCK'} (top={top_score:.3f} k={len(context_matches)}) -> "
                  f"Def={defense_layer:<14} Abstain={final_abstain}")

    finally:
        close_pool()

    # Tính toán báo cáo tổng hợp
    missing_records = [r for r in records if r["category"] == "in_scope_missing"]
    ans_records = [r for r in records if r["category"] != "in_scope_missing"]

    missing_total = len(missing_records)
    missing_retrieval_blocked = sum(not r["retrieval_passed"] for r in missing_records)
    missing_retrieval_leaked = missing_total - missing_retrieval_blocked
    missing_llm_blocked = sum(r["retrieval_passed"] and r["final_abstain"] for r in missing_records)
    missing_total_refused = sum(r["final_abstain"] for r in missing_records)
    missing_hallucinations = sum(not r["final_abstain"] for r in missing_records)

    ans_total = len(ans_records)
    ans_retrieval_false_refusals = sum(not r["retrieval_passed"] for r in ans_records)
    ans_llm_false_refusals = sum(r["retrieval_passed"] and r["final_abstain"] for r in ans_records)
    ans_answered = sum(not r["final_abstain"] for r in ans_records)
    ans_fragment_matches = sum(r["fragment_match"] is True for r in ans_records)

    report = {
        "model": "google_genai/gemini-3.5-flash-lite",
        "dataset": "data/rag_golden_set.json",
        "total_cases": len(records),
        "total_gemini_api_calls": total_calls,
        "summary": {
            "in_scope_missing": {
                "total_cases": missing_total,
                "retrieval_gate_blocked": missing_retrieval_blocked,
                "retrieval_gate_leaked": missing_retrieval_leaked,
                "retrieval_leakage_rate": round(missing_retrieval_leaked / missing_total, 3),
                "llm_generation_blocked": missing_llm_blocked,
                "total_end_to_end_refused": missing_total_refused,
                "end_to_end_abstention_recall": round(missing_total_refused / missing_total, 3),
                "true_hallucination_false_answers": missing_hallucinations,
                "hallucination_rate": round(missing_hallucinations / missing_total, 3),
            },
            "answerable": {
                "total_cases": ans_total,
                "retrieval_false_refusals": ans_retrieval_false_refusals,
                "llm_false_refusals": ans_llm_false_refusals,
                "total_answered": ans_answered,
                "expected_fragment_match": ans_fragment_matches,
                "fragment_match_rate": round(ans_fragment_matches / ans_total, 3),
            },
            "overall": {
                "abstention_accuracy": round(sum(r["abstention_correct"] for r in records) / len(records), 3),
            }
        },
        "records": records,
    }

    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved end-to-end production report to {OUTPUT_PATH}")
    print("\n" + "=" * 60)
    print("PRODUCTION RAG TWO-LAYER ABSTENTION EVALUATION (GEMINI 3.5 FLASH-LITE)")
    print("=" * 60)
    print(f"1. In-Scope Missing Data (N={missing_total}):")
    print(f"   - Retrieval Gate Blocked : {missing_retrieval_blocked}/{missing_total} ({missing_retrieval_blocked/missing_total:.1%})")
    print(f"   - Retrieval Gate Leaked  : {missing_retrieval_leaked}/{missing_total} ({missing_retrieval_leaked/missing_total:.1%}) [Cosine score overlap]")
    print(f"   - LLM Secondary Defense  : {missing_llm_blocked}/{missing_retrieval_leaked} leaked cases stopped by Gemini")
    print(f"   - End-to-End Refused     : {missing_total_refused}/{missing_total} ({missing_total_refused/missing_total:.1%})")
    print(f"   - Actual Hallucinations  : {missing_hallucinations}/{missing_total} ({missing_hallucinations/missing_total:.1%})")
    print(f"\n2. Answerable Cases (N={ans_total}):")
    print(f"   - Answered Successfully  : {ans_answered}/{ans_total} ({ans_answered/ans_total:.1%})")
    print(f"   - Fragment Match Rate    : {ans_fragment_matches}/{ans_total} ({ans_fragment_matches/ans_total:.1%})")
    print(f"   - False Abstentions      : {ans_retrieval_false_refusals + ans_llm_false_refusals}/{ans_total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
