"""Cross-check eval/current_numbers.md against the artifacts it cites.

The report is prose, so this checker targets the numbers that are easy to get
wrong after a rerun: abstention counts, retrieval variants, and the collected
test count. It exits non-zero with a precise message on the first mismatch.

Usage:
    python -m scripts.check_current_numbers
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NUMBERS_PATH = ROOT / "eval" / "current_numbers.md"
ABSTENTION_PATH = ROOT / "eval" / "dev_abstention_prompt_gemini35_m2.json"
VARIANTS_PATH = ROOT / "eval" / "rag_variants_dev.json"
ANSWER_VARIANTS_PATH = ROOT / "eval" / "rag_variants_answers_v2v4.json"
REPORT_PATH = ROOT / "eval" / "report_v2.md"


class CheckError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def require_text(text: str, snippet: str) -> None:
    require(snippet in text, f"Không thấy đoạn bắt buộc trong current_numbers.md: {snippet!r}")


def table_rows(text: str, first_cell: str) -> list[str]:
    prefix = f"| {first_cell} |"
    return [line for line in text.splitlines() if line.startswith(prefix)]


def table_row(text: str, first_cell: str, *, contains: str | None = None) -> str:
    rows = table_rows(text, first_cell)
    if contains is not None:
        rows = [row for row in rows if contains in row]
    if rows:
        return rows[0]
    suffix = f" chứa {contains!r}" if contains else ""
    raise CheckError(
        f"Không thấy dòng bảng bắt đầu bằng {first_cell!r}{suffix}"
    )


def check_abstention(text: str) -> None:
    summary = json.loads(ABSTENTION_PATH.read_text(encoding="utf-8"))["summary"]
    overall = summary["overall"]
    answerable = summary["answerable"]

    require_text(
        text,
        f"{overall['correct_cases']}/{overall['decided_cases']} = "
        f"{overall['abstention_accuracy']:.3f}",
    )
    rate = answerable["fragment_match_rate"] * 100
    require_text(
        text,
        f"{answerable['expected_fragment_match']}/{answerable['total_cases']} = {rate:.1f}%",
    )
    require_text(text, f"{summary['in_scope_missing']['true_hallucination_false_answers']}"
                        f"/{overall['expected_abstain_cases']}")

    rows = {
        "answerable": (
            answerable["total_cases"],
            answerable["retrieval_false_refusals"],
            answerable["retrieval_false_refusals"] + answerable["llm_false_refusals"],
        ),
        "in_scope_missing": (
            summary["in_scope_missing"]["total_cases"],
            summary["in_scope_missing"]["retrieval_gate_blocked"],
            summary["in_scope_missing"]["total_end_to_end_refused"],
        ),
        "out_of_scope_far": (
            summary["out_of_scope_far"]["total_cases"],
            summary["out_of_scope_far"]["retrieval_gate_blocked"],
            summary["out_of_scope_far"]["total_end_to_end_refused"],
        ),
    }
    gate_blocked = 0
    final_refused = 0
    for label, (total, blocked, refused) in rows.items():
        row = table_row(text, label)
        require(
            f"| {total} | {blocked} | {refused} |" in row,
            f"Dòng '{label}' không khớp artifact: cần | {total} | {blocked} | {refused} |",
        )
        gate_blocked += blocked
        final_refused += refused
    total_row = table_row(text, "**Tổng**")
    require(
        f"**{overall['total_cases']}** | **{gate_blocked}** | **{final_refused}**" in total_row,
        f"Dòng tổng không khớp artifact: cần **{overall['total_cases']}** | "
        f"**{gate_blocked}** | **{final_refused}**",
    )


def check_variants(text: str) -> None:
    variants = json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))["variants"]
    for name in ("V0", "V1", "V2", "V3", "V4"):
        metrics = variants[name]["metrics"]
        hit1 = metrics["hit@1"]
        hit1_cell = (
            f"{hit1:.4f} ({round(hit1 * metrics['answerable_cases'])}/"
            f"{metrics['answerable_cases']})"
        )
        # Biến thể xuất hiện ở cả bảng retrieval lẫn bảng lượt trả lời; chọn
        # đúng dòng retrieval bằng ô Hit@1 có 4 chữ số thập phân.
        row = table_row(text, name, contains=hit1_cell)
        expected = [
            hit1_cell,
            f"{metrics['hit@5']:.4f}",
            f"{metrics['mrr']:.4f}",
            str(metrics["answerable_retrieval_false_refusals"]),
            f"{metrics['negative_gate_leaks']}/{metrics['negative_cases']}",
        ]
        for value in expected:
            require(value in row, f"Dòng {name} thiếu giá trị {value!r} từ artifact")


def check_answer_variants(text: str) -> None:
    """Đối chiếu bảng lượt trả lời với artifact V0 và V2/V3/V4."""
    v0 = json.loads(
        (ROOT / "eval" / "rag_variants_answers_v0_run1.json").read_text(encoding="utf-8")
    )["variants"]["V0"]["answer_metrics"]
    measured = json.loads(ANSWER_VARIANTS_PATH.read_text(encoding="utf-8"))["variants"]

    def row_for(label: str, cells: list[str]) -> str:
        rows = table_rows(text, label)
        for row in rows:
            if all(cell in row for cell in cells):
                return row
        raise CheckError(f"Không thấy dòng '{label}' chứa {cells}")

    expected = {
        "V2": measured["V2"]["answer_metrics"],
        "V3": measured["V3"]["answer_metrics"],
    }
    for label, metrics in expected.items():
        answered = f"{metrics['answerable_answered']}/{metrics['answerable_total']}"
        fragments = f"{metrics['answerable_fragment_matches']}/{metrics['answerable_total']}"
        hallucinations = f"{metrics['negative_hallucinated_answers']}/{metrics['negative_total']}"
        row = row_for(label, [answered, hallucinations])
        require(
            fragments in row,
            f"Dòng lượt trả lời {label} thiếu fragment {fragments!r}",
        )

    answers = f"{v0['answerable_answered']}/{v0['answerable_total']}"
    row = row_for("V0", [answers])
    require(
        f"{v0['answerable_fragment_matches']}/{v0['answerable_total']}" in row,
        "Dòng lượt trả lời V0 thiếu fragment_match từ artifact",
    )

    v4 = measured["V4"]["answer_metrics"]
    decided = v4["answerable_decided"]
    row = row_for(
        "V4",
        [
            f"{v4['answerable_answered']}/{decided} ca quyết định",
            f"{v4['negative_hallucinated_answers']}/{v4['negative_decided']} ca quyết định",
        ],
    )
    require(
        f"{v4['answerable_fragment_matches']}/{decided} ca quyết định" in row,
        "Dòng lượt trả lời V4 thiếu fragment_match theo số ca quyết định",
    )


def check_test_count(text: str) -> None:
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    match = re.search(r"(\d+) tests collected", collected)
    require(match is not None, "Không đọc được số test thu thập từ pytest")
    count = int(match.group(1))
    require_text(text, f"**{count}** test")
    report = REPORT_PATH.read_text(encoding="utf-8")
    require_text(report, f"**{count} tests**")


def main() -> int:
    text = NUMBERS_PATH.read_text(encoding="utf-8")
    try:
        check_abstention(text)
        check_variants(text)
        check_answer_variants(text)
        check_test_count(text)
    except CheckError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "OK: current_numbers.md khớp artifact abstention, biến thể retrieval, "
        "lượt trả lời và số test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
