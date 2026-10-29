"""
Tự động sinh bảng kiểm thử Markdown cho Mục 7 của eval/report_v2.md
dựa trên kết quả thu thập thực tế từ `pytest --collect-only -q`.
Ngăn chặn hoàn toàn việc gõ tay sai lệch số lượng test giữa code và báo cáo.
"""

import subprocess
import sys
from pathlib import Path


def generate_markdown_test_table() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Lỗi khi thu thập test: {res.stderr}")

    file_counts = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if "::" in line:
            fp = line.split("::")[0].replace("\\", "/")
            file_counts[fp] = file_counts.get(fp, 0) + 1

    total_tests = sum(file_counts.values())
    total_files = len(file_counts)

    lines = []
    lines.append("| STT | File Kiểm Thử | Thu thập (Collected) | Trạng thái Chạy | Mô Tả & Phạm Vi Kiểm Thử |")
    lines.append("| :---: | :--- | :---: | :---: | :--- |")

    descriptions = {
        "tests/test_api_client.py": "Xác thực `X-API-Key`, mã lỗi 401, 422, mock RAG/NLQ/Router/Feedback.",
        "tests/test_chat_history.py": "Quản lý session LRU, dọn corpus nền không chặn event loop, quét corpus mồ côi, cắt tỉa ngữ cảnh hội thoại.",
        "tests/test_chunking.py": "Chiến lược phân đoạn heading-bounded, fixed-size, bảo toàn tiếng Việt Unicode NFC.",
        "tests/test_evaluate_rag_answers.py": "Đánh giá chất lượng sinh câu trả lời RAG.",
        "tests/test_evaluate_rag_production.py": "Đo đạc kiểm thử pipeline RAG với mô hình production.",
        "tests/test_evaluators.py": "Đánh giá SQL guard, so sánh kết quả bảng, cơ chế retry khi lỗi.",
        "tests/test_golden_data_validation.py": "Kiểm tra tính toàn vẹn và phân bổ nhãn của tập golden set 65 ca.",
        "tests/test_health.py": "Liveness, readiness probes, kiểm tra kết nối Postgres và Ollama.",
        "tests/test_ingestion.py": "Trích xuất văn bản từ docx, pdf, xử lý lỗi định dạng và ký tự đặc biệt.",
        "tests/test_integration_postgres.py": "Kiểm thử live database pgvector, cô lập corpus session A/B và phân quyền `ai_nlq_reader` (chạy trên CI/Live DB).",
        "tests/test_llm.py": "Phân giải vai trò model (fast/strong/local), cache instance mô hình.",
        "tests/test_nlq_runtime.py": "NLQ graph sinh SQL, validator, test với `FakeListChatModel`, mã lỗi 503/504.",
        "tests/test_prompt_and_streaming.py": "Định dạng SSE, bắt sự kiện `start`, `sources`, `token`, `no_answer`, `error`.",
        "tests/test_rag.py": "Cosine gating, từ chối khi thiếu context, test với `FakeListChatModel` (JSON chuẩn, JSON hỏng, abstain).",
        "tests/test_rag_heldout.py": "Đánh giá độ trễ và tỷ lệ recall trên tập held-out M1/M2.",
        "tests/test_rag_upload.py": "Endpoint upload 5MB limit, byte-counting middleware (413), cô lập corpus theo session, NamedTemporaryFile tự hủy.",
        "tests/test_report_v2_metrics.py": "Kiểm tra tự động đảm bảo số liệu báo cáo khớp 100% với dữ liệu nguồn và test table.",
        "tests/test_response_contracts.py": "Chuẩn hóa schema Pydantic, xác thực trường `trace_id` trong response.",
        "tests/test_retriever.py": "Tìm kiếm vector, BM25 cache theo corpus có giới hạn, hybrid fusion, reranker.",
        "tests/test_router.py": "Ma trận phân loại 2 trục, đánh giá tập đóng băng 70 câu, test với `FakeListChatModel`.",
        "tests/test_security.py": "Rate limiting theo IP, chống timing attack, `MaxBodySizeMiddleware`.",
        "tests/test_tracing.py": "Context manager tracing, an toàn khi Langfuse ngoại tuyến, sinh pre-generated `trace_id`.",
        "tests/test_vector_store.py": "Cosine distance, thay thế corpus, upsert và tìm kiếm theo danh sách `corpus = ANY(%s)`.",
    }

    for idx, (fp, count) in enumerate(sorted(file_counts.items()), 1):
        desc = descriptions.get(fp, "Kiểm thử tự động các thành phần hệ thống.")
        status_str = "Chờ artifact JUnit trên CI"
        lines.append(f"| {idx} | `{fp}` | **{count}** | {status_str} | {desc} |")

    lines.append("")
    lines.append(
        f"👉 **Tổng số kiểm thử thu thập:** **{total_tests} tests** "
        f"trên toàn bộ **{total_files} file test** "
        "(số lượng thu thập, không phải số lượng đã chạy/đạt)."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(generate_markdown_test_table())
