"""
End-to-end production measurement of RAG abstention using Google Gemini 3.5 Flash-Lite.

Measures both the Retrieval Gate (Layer 1) and LLM Grounding Defense (Layer 2)
across all 65 development cases: 41 answerable, 19 `in_scope_missing` and
5 `out_of_scope_far`. Both negative categories are expected abstentions, per
`docs/rag_eval_protocol.md` ("false positive means retrieval returns a result
above the configured cutoff for a case that should be refused").

Strict 4.2s pacing to stay safely under 15 RPM. The pacing sleep happens before
the generation timer starts, so `latency_ms` measures generation only.
"""

import json
from pathlib import Path
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):  # pytest capture objects may not support it
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import settings
from app.core.llm import MODEL_REGISTRY, get_llm
from app.core.rag_prompt import (
    NO_ANSWER,
    build_rag_chain,
    format_context,
)
from app.core.vector_store import close_pool, similarity_search

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data/rag_golden_set.json"
OUTPUT_PATH = ROOT / "eval/dev_abstention_prompt_gemini35_m1.json"

# Categories the system is expected to refuse. See docs/rag_eval_protocol.md.
ABSTAIN_CATEGORIES = frozenset({"in_scope_missing", "out_of_scope_far"})
# Protocol: suppress rates for groups with fewer than this many cases.
MIN_CASES_FOR_RATE = 10


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


def _negative_group(group: list[dict]) -> dict:
    """Aggregate one expected-abstain category.

    Errored LLM calls are undecided: they are reported separately and excluded
    from recall/hallucination rates instead of counting as answered.
    """
    total = len(group)
    blocked = sum(not r["retrieval_passed"] for r in group)
    leaked = total - blocked
    usable = [r for r in group if not r.get("errored")]
    decided = len(usable)
    refused = sum(r["final_abstain"] for r in usable)
    hallucinations = sum(not r["final_abstain"] for r in usable)
    row = {
        "total_cases": total,
        "error_cases": total - decided,
        "retrieval_gate_blocked": blocked,
        "retrieval_gate_leaked": leaked,
        "retrieval_leakage_rate": round(leaked / total, 3) if total else None,
        "llm_generation_blocked": sum(
            r["retrieval_passed"] and r["final_abstain"] for r in usable
        ),
        "total_end_to_end_refused": refused,
        "end_to_end_abstention_recall": round(refused / decided, 3) if decided else None,
        "true_hallucination_false_answers": hallucinations,
        "hallucination_rate": round(hallucinations / decided, 3) if decided else None,
    }
    if decided < MIN_CASES_FOR_RATE:
        row["rate_note"] = f"Fewer than {MIN_CASES_FOR_RATE} decided cases; counts only."
        for key in (
            "end_to_end_abstention_recall",
            "hallucination_rate",
        ):
            row[key] = None
    return row


def summarize_records(records: list[dict]) -> dict:
    """Build the report summary.

    Both `in_scope_missing` and `out_of_scope_far` count as expected abstentions,
    so refusing a far question is correct, not a false refusal.
    """
    missing = [r for r in records if r["category"] == "in_scope_missing"]
    far = [r for r in records if r["category"] == "out_of_scope_far"]
    answerable = [r for r in records if r["category"] not in ABSTAIN_CATEGORIES]

    ans_usable = [r for r in answerable if not r.get("errored")]
    ans_total = len(answerable)
    ans_answered = sum(not r["final_abstain"] for r in ans_usable)
    ans_fragment_matches = sum(r["fragment_match"] is True for r in ans_usable)

    expected_abstain = missing + far
    expected_abstain_refused = sum(
        r["final_abstain"] for r in expected_abstain if not r.get("errored")
    )
    decided = [r for r in records if not r.get("errored")]
    # Recompute correctness from the label rule instead of trusting a stored field.
    correct = sum(
        bool(r["final_abstain"]) == (r["category"] in ABSTAIN_CATEGORIES)
        for r in decided
    )

    return {
        "in_scope_missing": _negative_group(missing),
        "out_of_scope_far": _negative_group(far),
        "answerable": {
            "total_cases": ans_total,
            "error_cases": ans_total - len(ans_usable),
            "retrieval_false_refusals": sum(not r["retrieval_passed"] for r in ans_usable),
            "llm_false_refusals": sum(
                r["retrieval_passed"] and r["final_abstain"] for r in ans_usable
            ),
            "total_answered": ans_answered,
            "expected_fragment_match": ans_fragment_matches,
            "fragment_match_rate": (
                round(ans_fragment_matches / len(ans_usable), 3) if ans_usable else None
            ),
        },
        "overall": {
            "abstention_accuracy": round(correct / len(decided), 3) if decided else None,
            "correct_cases": correct,
            "decided_cases": len(decided),
            "error_cases": len(records) - len(decided),
            "total_cases": len(records),
            "expected_abstain_cases": len(expected_abstain),
            "expected_abstain_refused": expected_abstain_refused,
            "answerable_total": ans_total,
            "answerable_answered": ans_answered,
        },
    }


def main(output_path: Path = OUTPUT_PATH, model: str = "fast", pacing_seconds: float = 4.2):
    print(f"Loading golden dataset from {GOLDEN_PATH}...")
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"Total cases: {len(cases)}")

    # Nhóm ca kiểm thử: 41 answerable, 19 in_scope_missing, 5 out_of_scope_far
    total_cases = len(cases)
    missing_cases = [c for c in cases if c.get("category") == "in_scope_missing"]
    far_cases = [c for c in cases if c.get("category") == "out_of_scope_far"]
    answerable_cases = [c for c in cases if c.get("category") not in ABSTAIN_CATEGORIES]
    print(f"  - Answerable cases    : {len(answerable_cases)}")
    print(f"  - In-scope missing    : {len(missing_cases)}")
    print(f"  - Out-of-scope far    : {len(far_cases)}")

    llm = get_llm(model, temperature=0)
    chain = build_rag_chain(llm)

    records = []
    total_calls = 0

    try:
        # Chạy toàn bộ dataset (65 ca hiện hành)
        for idx, case in enumerate(cases, 1):
            cid = case["id"]
            cat = case.get("category", "simple")
            expected_abstain = cat in ABSTAIN_CATEGORIES

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
                latency_ms = None  # Gate chặn: không gọi LLM nên không có độ trễ sinh
            else:
                context = format_context(context_matches)
                time.sleep(pacing_seconds)  # Pacing to stay under the provider rate limit
                started = time.perf_counter()
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
            errored = model_abstain is None
            abstention_correct = (
                None if errored else (final_abstain == expected_abstain)
            )

            parts = case.get("answer_contains", [])
            if isinstance(parts, str):
                parts = [parts]
            fragment_match = (
                all(normalize(part) in normalize(answer) for part in parts)
                if not expected_abstain and not final_abstain and answer != NO_ANSWER else None
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
                "errored": errored,
                "fragment_match": fragment_match,
                "defense_layer": defense_layer,
                "latency_ms": latency_ms,
            }
            records.append(record)

            status_icon = "✅" if abstention_correct else ("⚠️" if errored else "❌")
            top_display = f"{top_score:.3f}" if top_score is not None else "n/a"
            print(f"[{idx:02d}/{total_cases}] {status_icon} {cid} ({cat:<17}): "
                  f"Gate={'PASS' if retrieval_passed else 'BLOCK'} (top={top_display} k={len(context_matches)}) -> "
                  f"Def={defense_layer:<14} Abstain={final_abstain}")

    finally:
        close_pool()

    # Tính toán báo cáo tổng hợp
    summary = summarize_records(records)

    report = {
        "model": MODEL_REGISTRY.get(model, model),
        "model_role": model,
        "dataset": "data/rag_golden_set.json",
        "total_cases": len(records),
        "total_gemini_api_calls": total_calls,
        "pacing_seconds": pacing_seconds,
        "abstention_policy": (
            "in_scope_missing and out_of_scope_far are both expected abstentions "
            "(docs/rag_eval_protocol.md)."
        ),
        "latency_field_note": (
            "latency_ms covers LLM generation only: the 4.2s pacing sleep happens "
            "before the timer starts, and gate-blocked cases report null."
        ),
        "summary": summary,
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved end-to-end production report to {output_path}")
    print("\n" + "=" * 60)
    print(f"PRODUCTION RAG TWO-LAYER ABSTENTION EVALUATION ({report['model']})")
    print("=" * 60)

    def _print_negative(title: str, group: dict) -> None:
        total = group["total_cases"]
        leaked = group["retrieval_gate_leaked"]
        print(f"{title} (N={total}):")
        print(f"   - Retrieval Gate Blocked : {group['retrieval_gate_blocked']}/{total}")
        print(f"   - Retrieval Gate Leaked  : {leaked}/{total} [Cosine score overlap]")
        print(f"   - LLM Secondary Defense  : {group['llm_generation_blocked']}/{leaked} leaked cases stopped by Gemini")
        print(f"   - End-to-End Refused     : {group['total_end_to_end_refused']}/{total}")
        print(f"   - Actual Hallucinations  : {group['true_hallucination_false_answers']}/{total}")
        if group.get("error_cases"):
            print(f"   - Errored (undecided)    : {group['error_cases']}/{total}")
        if group.get("rate_note"):
            print(f"   - Note                   : {group['rate_note']}")

    ans = summary["answerable"]
    overall = summary["overall"]
    decided = overall["decided_cases"]

    _print_negative("1. In-Scope Missing Data", summary["in_scope_missing"])
    print()
    _print_negative("2. Out-of-Scope Far", summary["out_of_scope_far"])
    print(f"\n3. Answerable Cases (N={ans['total_cases']}):")
    usable = ans["total_cases"] - ans["error_cases"]
    print(f"   - Answered Successfully  : {ans['total_answered']}/{usable}")
    print(f"   - Fragment Match Rate    : {ans['expected_fragment_match']}/{usable}")
    print(f"   - False Abstentions      : {ans['retrieval_false_refusals'] + ans['llm_false_refusals']}/{usable}")
    if ans["error_cases"]:
        print(f"   - Errored (undecided)    : {ans['error_cases']}/{ans['total_cases']}")
    print("\n4. Overall:")
    print(f"   - Expected Abstain Groups: {overall['expected_abstain_cases']} ca, refused {overall['expected_abstain_refused']}")
    if decided:
        print(f"   - Abstention Accuracy    : {round(overall['abstention_accuracy'] * 100, 1)}% ({overall['correct_cases']}/{decided} decided)")
    else:
        print("   - Abstention Accuracy    : n/a (không có ca quyết định được)")
    if overall["error_cases"]:
        print(f"   - Errored (excluded)     : {overall['error_cases']}/{overall['total_cases']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Đường dẫn artifact. Đừng ghi đè artifact M1 đã đóng băng.",
    )
    parser.add_argument(
        "--model",
        default="fast",
        help="Role (fast/strong/local) hoặc model string, ví dụ google_genai/gemini-3.6-flash.",
    )
    parser.add_argument(
        "--pacing",
        type=float,
        default=4.2,
        help="Giây nghỉ giữa các lần gọi để tránh rate limit (không tính vào latency).",
    )
    args = parser.parse_args()
    main(
        args.output if args.output.is_absolute() else ROOT / args.output,
        args.model,
        args.pacing,
    )
