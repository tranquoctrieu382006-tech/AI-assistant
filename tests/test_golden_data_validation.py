import hashlib
import json
import unicodedata
from pathlib import Path

from scripts.rag_eval_metrics import context_recall_at_k, paired_bootstrap_interval
from scripts.validate_golden_data import validate_rag_cases, validate_text
from scripts.evaluate_rag_retrieval import _abstention_risk_summary
from scripts.manifest_hashing import sha256_file
from scripts.measure_production_abstention_gemini import summarize_records
from scripts.report_manifest_drift import collect


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_rag_golden_data_and_evidence_are_valid():
    cases = json.loads((ROOT / "data/rag_golden_set.json").read_text(encoding="utf-8"))
    assert validate_rag_cases(cases) == []


def test_text_validator_detects_question_mark_between_letters_but_not_sentence_end():
    assert validate_text("xin?nghỉ", "q")
    assert validate_text("Xin nghỉ phép?", "q") == []


def test_text_validator_requires_unicode_nfc():
    decomposed = unicodedata.normalize("NFD", "tiếng Việt")
    assert any("NFC" in error for error in validate_text(decomposed, "q"))


def test_manifest_hash_normalizes_text_line_endings_but_not_binary(tmp_path):
    lf = tmp_path / "a.md"
    crlf = tmp_path / "b.md"
    lf.write_bytes(b"one\ntwo\n")
    crlf.write_bytes(b"one\r\ntwo\r\n")
    assert sha256_file(lf) == sha256_file(crlf)

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"binary\n")
    pdf_b.write_bytes(b"binary\r\n")
    assert sha256_file(pdf_a) != sha256_file(pdf_b)


def test_context_recall_matches_evidence_split_across_adjacent_chunks():
    case = {"evidence": [{"source": "policy.md", "text": "bảy ngày lịch"}]}
    results = [
        {"source": "policy.md", "chunk_index": 4, "content": "Thời hạn đổi trả là bảy ngày"},
        {"source": "policy.md", "chunk_index": 5, "content": "lịch kể từ ngày giao thành công"},
    ]
    assert context_recall_at_k(case, results, 2) == 1.0


def test_context_recall_does_not_join_non_adjacent_chunks_or_other_sources():
    case = {"evidence": [{"source": "policy.md", "text": "bảy ngày lịch"}]}
    results = [
        {"source": "policy.md", "chunk_index": 4, "content": "Thời hạn là bảy ngày"},
        {"source": "policy.md", "chunk_index": 8, "content": "lịch sau khi giao"},
        {"source": "other.md", "chunk_index": 5, "content": "bảy ngày lịch"},
    ]
    assert context_recall_at_k(case, results, 3) == 0.0


def test_context_recall_only_counts_results_inside_k():
    case = {"evidence": [{"source": "policy.md", "text": "bảy ngày lịch"}]}
    results = [
        {"source": "other.md", "chunk_index": 0, "content": "khác"},
        {"source": "policy.md", "chunk_index": 1, "content": "bảy ngày lịch"},
    ]
    assert context_recall_at_k(case, results, 1) == 0.0


def test_abstention_risk_reports_counts_only_below_ten_cases():
    summary = _abstention_risk_summary([
        {"category": "in_scope_missing", "correct": False},
        {"category": "in_scope_missing", "correct": True},
        {"category": "out_of_scope_far", "correct": True},
    ])
    assert summary["in_scope_missing"]["false_positive_count"] == 1
    assert summary["in_scope_missing"]["false_positive_rate"] is None
    assert summary["out_of_scope_far"]["false_positive_rate"] is None


def test_paired_bootstrap_is_reproducible_and_uses_matched_scores():
    first = paired_bootstrap_interval([0, 1, 1, 0], [1, 1, 0, 0], samples=500)
    second = paired_bootstrap_interval([0, 1, 1, 0], [1, 1, 0, 0], samples=500)
    assert first == second
    assert first["estimate"] == 0
    assert first["paired_cases"] == 4
    assert first["lower"] <= first["estimate"] <= first["upper"]


def test_manifest_drift_check_is_line_ending_portable_and_flags_content_drift(tmp_path):
    text = tmp_path / "policy.md"
    text.write_bytes(b"one\r\ntwo\r\n")
    changed = tmp_path / "changed.md"
    changed.write_bytes(b"amended\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest": "T",
                "corpus_files": {
                    "policy.md": hashlib.sha256(b"one\ntwo\n").hexdigest(),
                    "changed.md": hashlib.sha256(b"original\n").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    rows = {row["path"]: row for row in collect(manifest, root=tmp_path)}
    assert rows["policy.md"]["status"] == "ok"
    assert rows["changed.md"]["status"] == "drift"
    assert rows["changed.md"]["line_endings"] == "lf"


def test_abstention_summary_treats_far_as_expected_abstain():
    def record(category, retrieval_passed, final_abstain, correct, fragment=None):
        return {
            "category": category,
            "retrieval_passed": retrieval_passed,
            "final_abstain": final_abstain,
            "abstention_correct": correct,
            "fragment_match": fragment,
        }

    records = [
        record("simple", True, False, True, True),
        record("in_scope_missing", False, True, True),
        record("out_of_scope_far", False, True, True),
    ]

    summary = summarize_records(records)

    # Từ chối câu out_of_scope_far là đúng, không phải false refusal.
    assert summary["overall"]["abstention_accuracy"] == 1.0
    assert summary["overall"]["expected_abstain_cases"] == 2
    assert summary["overall"]["expected_abstain_refused"] == 2
    assert summary["overall"]["answerable_total"] == 1
    assert summary["out_of_scope_far"]["total_end_to_end_refused"] == 1
    # Nhóm 5 ca far: chỉ báo count, không báo rate.
    assert summary["out_of_scope_far"]["end_to_end_abstention_recall"] is None
    assert summary["out_of_scope_far"]["rate_note"] is not None


def test_abstention_summary_excludes_errored_calls_from_answered_and_accuracy():
    def record(category, retrieval_passed, final_abstain, errored=False, fragment=None):
        return {
            "category": category,
            "retrieval_passed": retrieval_passed,
            "final_abstain": final_abstain,
            "errored": errored,
            "fragment_match": fragment,
        }

    records = [
        record("simple", True, False, fragment=True),
        record("simple", True, False, errored=True),   # 429, chưa biết kết quả
        record("in_scope_missing", True, False, errored=True),
    ]

    summary = summarize_records(records)

    # Ca lỗi không được tính là "đã trả lời" và bị loại khỏi mẫu số accuracy.
    assert summary["answerable"]["total_answered"] == 1
    assert summary["answerable"]["error_cases"] == 1
    assert summary["overall"]["error_cases"] == 2
    assert summary["overall"]["decided_cases"] == 1
    assert summary["overall"]["abstention_accuracy"] == 1.0
    assert summary["in_scope_missing"]["true_hallucination_false_answers"] == 0


def test_multi_document_case_must_use_two_distinct_sources(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("Evidence passage.", encoding="utf-8")
    case = {
        "id": "multi-001",
        "question": "Combine two sources?",
        "answer_contains": ["Evidence passage."],
        "source": "policy.md",
        "expected_sources": ["policy.md"],
        "category": "multi_document",
        "evidence": [{"source": "policy.md", "text": "Evidence passage."}],
    }
    errors = validate_rag_cases([case], corpus_dir=tmp_path)
    assert any("at least two sources" in error for error in errors)
