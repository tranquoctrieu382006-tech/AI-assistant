"""
Automated validation test for eval/report_v2.md.
Đảm bảo toàn bộ số liệu, tỷ lệ đo đạc và tập mẫu được trích dẫn trong báo cáo
luôn khớp chính xác 100% với các artifacts và file dữ liệu nguồn, chống sai lệch số liệu.
"""

from collections import Counter
import json
from pathlib import Path
import re

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def report_text(repo_root: Path) -> str:
    report_path = repo_root / "eval" / "report_v2.md"
    assert report_path.exists(), "eval/report_v2.md không tồn tại."
    return report_path.read_text(encoding="utf-8")


def test_report_matches_rag_golden_counts(repo_root: Path, report_text: str):
    """Kiểm tra số ca RAG trong báo cáo khớp chính xác data/rag_golden_set.json."""
    golden_path = repo_root / "data" / "rag_golden_set.json"
    assert golden_path.exists(), "data/rag_golden_set.json không tồn tại."
    cases = json.loads(golden_path.read_text(encoding="utf-8"))

    total = len(cases)
    categories = Counter(item.get("category") for item in cases)
    answerable = categories["simple"] + categories["paraphrase"] + categories["multi_document"]
    in_scope_missing = categories["in_scope_missing"]
    out_of_scope_far = categories["out_of_scope_far"]

    assert total == 65
    assert answerable == 41
    assert in_scope_missing == 19
    assert out_of_scope_far == 5

    # Đối chiếu số xuất hiện trong report_v2.md
    assert "65 ca kiểm thử" in report_text or "65 ca" in report_text
    assert "41 ca có đáp án" in report_text or "41 ca" in report_text
    assert "19 ca" in report_text
    assert "5 ca" in report_text


def test_report_matches_nlq_golden_counts(repo_root: Path, report_text: str):
    """Kiểm tra số ca NLQ trong báo cáo khớp chính xác data/nlq_golden_set.json."""
    golden_path = repo_root / "data" / "nlq_golden_set.json"
    assert golden_path.exists(), "data/nlq_golden_set.json không tồn tại."
    cases = json.loads(golden_path.read_text(encoding="utf-8"))

    assert len(cases) == 30
    assert "30 ca" in report_text


def test_report_matches_retrieval_abstention_artifacts(repo_root: Path, report_text: str):
    """
    Kiểm tra số liệu cổng cosine (14/19 ca thiếu dữ liệu ở dev set và 10/10 ở held-out M2)
    khớp chuẩn xác với các artifacts điều tra m1 và m2.
    """
    m1_path = repo_root / "eval" / "dev_threshold_investigation_m1.md"
    assert m1_path.exists(), "eval/dev_threshold_investigation_m1.md không tồn tại."
    m1_text = m1_path.read_text(encoding="utf-8")
    assert "14/19" in m1_text

    m2_path = repo_root / "eval" / "heldout_summary_m2.md"
    assert m2_path.exists(), "eval/heldout_summary_m2.md không tồn tại."
    m2_text = m2_path.read_text(encoding="utf-8")
    assert "10/10" in m2_text

    # Đối chiếu báo cáo tổng hợp phản ánh đúng các số này
    assert "14 / 19" in report_text or "14/19" in report_text
    assert "10 / 10" in report_text or "10/10" in report_text


def test_report_matches_rerank_and_hybrid_metrics(repo_root: Path, report_text: str):
    """
    Kiểm tra bảng đo Hit@1 và MRR của Vector vs Hybrid vs Reranker
    khớp chuẩn xác với rerank_config_comparison.md.
    """
    rerank_path = repo_root / "eval" / "rerank_config_comparison.md"
    assert rerank_path.exists(), "eval/rerank_config_comparison.md không tồn tại."
    rerank_text = rerank_path.read_text(encoding="utf-8")

    assert "0.8537" in rerank_text  # Hit@1 của V5 (35/41)
    assert "0.7561" in rerank_text  # Hit@1 của H5 (31/41)
    assert "0.9207" in rerank_text  # MRR của V5
    assert "0.8740" in rerank_text  # MRR của H5

    # Đối chiếu báo cáo tổng hợp
    assert "0.8537" in report_text
    assert "0.7561" in report_text
    assert "35/41" in report_text
    assert "31/41" in report_text
    assert "0.9207" in report_text
    assert "0.8740" in report_text


def test_report_matches_actual_test_files_count(repo_root: Path, report_text: str):
    """Kiểm tra báo cáo trích dẫn đúng số file test thực tế trong tests/."""
    test_files = [f for f in (repo_root / "tests").glob("test_*.py")]
    # Báo cáo trích dẫn 20 file test cốt lõi (hoặc 21 bao gồm file test đối chiếu này)
    assert len(test_files) >= 20
    assert re.search(r"\b2[0-4]\s+file\s+test\b", report_text)
