import json
import unicodedata
from pathlib import Path

from scripts.rag_eval_metrics import context_recall_at_k, paired_bootstrap_interval
from scripts.validate_golden_data import validate_rag_cases, validate_text
from scripts.evaluate_rag_retrieval import _abstention_risk_summary


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
