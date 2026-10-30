"""
Automated validation test for eval/report_v2.md.
Đảm bảo toàn bộ số liệu, tỷ lệ đo đạc và tập mẫu được trích dẫn trong báo cáo
luôn khớp chính xác 100% với các artifacts và file dữ liệu nguồn, chống sai lệch số liệu.
"""

from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

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
    assert "v2_nlq_eval_run_20261028.json" in report_text


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
    khớp chuẩn xác với rerank_config_comparison.md và chuẩn hóa định dạng độ trễ.
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

    # Độ trễ phải ghi rõ đơn vị, tránh "2.571 ms" mơ hồ giữa dấu thập phân
    # và dấu phân cách hàng nghìn: 2571 ms = 2,571 s.
    assert "2.571.0" not in report_text, "Phát hiện định dạng lỗi 2.571.0 trong báo cáo"
    assert "4.641.1" not in report_text, "Phát hiện định dạng lỗi 4.641.1 trong báo cáo"
    assert "2571 ms" in report_text and "≈2,57 s" in report_text
    assert "4641 ms" in report_text and "≈4,64 s" in report_text


def test_report_matches_actual_test_files_count(repo_root: Path, report_text: str):
    """
    Kiểm tra Mục 7 của báo cáo khớp với pytest thu thập thực tế:
    - Đúng toàn bộ file test vật lý (không thiếu, không thừa), không hardcode 23/252.
    - Khớp chính xác số lượng test từng file được thu thập.
    """
    # 1. Kiểm tra không có test_limits.py trong repo và báo cáo
    assert not (repo_root / "tests" / "test_limits.py").exists(), "test_limits.py không được tồn tại trên đĩa."
    assert "test_limits.py" not in report_text, "Báo cáo vẫn chứa file phantom test_limits.py."

    # 2. Lấy danh sách file test thực tế trên đĩa
    test_files_on_disk = sorted(f.name for f in (repo_root / "tests").glob("test_*.py"))

    # 3. Thu thập số test thực tế từ pytest --collect-only
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode == 0, f"Lỗi thu thập pytest: {res.stderr}"

    actual_counts = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if "::" in line:
            fp = line.split("::")[0].replace("\\", "/")
            fname = Path(fp).name
            actual_counts[fname] = actual_counts.get(fname, 0) + 1

    total_collected = sum(actual_counts.values())
    total_files = len(actual_counts)

    # 4. Parse bảng Markdown Mục 7 từ report_v2.md
    # Định dạng mỗi dòng: | STT | `tests/test_xxx.py` | **N** | Status | Desc |
    table_pattern = re.compile(r"\|\s*\d+\s*\|\s*`tests/(test_[a-zA-Z0-9_]+\.py)`\s*\|\s*\*\*(\d+)\*\*\s*\|")
    matches = table_pattern.findall(report_text)

    reported_files = {m[0]: int(m[1]) for m in matches}
    assert len(reported_files) == total_files, (
        f"Bảng Mục 7 có {len(reported_files)} file, kỳ vọng {total_files} file."
    )

    for fname in test_files_on_disk:
        assert fname in reported_files, f"File {fname} thiếu trong bảng Mục 7."
        assert reported_files[fname] == actual_counts[fname], (
            f"Số test cho {fname} không khớp: báo cáo={reported_files[fname]}, thực tế={actual_counts[fname]}"
        )

    # 5. Kiểm tra tổng số trích dẫn trong văn bản (động theo lần thu thập)
    assert f"{total_collected} tests" in report_text
    assert f"{total_files} file test" in report_text


def test_report_does_not_claim_unverified_pass_or_human_audit(report_text: str):
    """Không được tuyên bố pass 100% hoặc Human Audit khi chưa có bằng chứng."""
    assert "100% Pass" not in report_text
    assert "100% consensus" not in report_text
    assert "Human Audit" not in report_text
    assert "chưa có thẩm định độc lập từ con người" in report_text
    assert "artifact JUnit" in report_text

    # 6. Kiểm tra các mục kiến trúc bổ sung
    assert "Soft Session Isolation" in report_text or "Phân vùng Corpus người dùng" in report_text
    assert "LLM-as-judge" in report_text
