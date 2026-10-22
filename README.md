# AI Assistant

Mini AI Assistant – Hỏi đáp tài liệu (RAG) & truy vấn dữ liệu bằng ngôn ngữ tự nhiên (NLQ).

## Tech Stack

- **Python 3.11+**, **FastAPI**, **LangChain + LangGraph**
- **Langfuse** (tracing, prompt management, eval)
- **PostgreSQL + pgvector** (vector store)
- **Ollama** (local LLM)

## Quick Start

```bash
# 1. Clone & cài đặt
git clone <repo-url>
cd ai-assistant
pip install -r requirements.txt

# 2. Cấu hình
cp .env.example .env
# Sửa .env: điền API key, chọn model

# 3. Chạy
uvicorn app.main:app --reload

# 4. Mở Swagger UI
# http://localhost:8000/docs
```

## Đổi mô hình

Sửa biến trong `.env`, **không cần sửa code**:

| Vai trò | Biến môi trường | Ví dụ |
|---|---|---|
| Mặc định | `DEFAULT_MODEL` | `openai/gpt-4o-mini` |
| Mạnh (SQL phức tạp, tổng hợp) | `STRONG_MODEL` | `openai/gpt-4o` |
| Nhanh/rẻ (router, viết lại câu hỏi) | `FAST_MODEL` | `openai/gpt-4o-mini` |
| Local (không gửi dữ liệu ra ngoài) | `LOCAL_MODEL` | `ollama/gemma3:12b` |

Groq có thể dùng với `GROQ_API_KEY` trong `.env`, ví dụ `groq/openai/gpt-oss-120b` và `groq/openai/gpt-oss-20b`. Gemini vẫn được hỗ trợ bằng `GOOGLE_API_KEY` và model có tiền tố `google_genai/`.

## Tại sao chọn `init_chat_model` thay vì LiteLLM?

| Tiêu chí | init_chat_model (LangChain) | LiteLLM |
|---|---|---|
| Dependency | Không thêm package | Thêm `litellm` |
| Tương thích LangChain | Native, luôn đồng bộ version | Đôi khi lag version |
| Số provider hỗ trợ | ~10 provider chính | 100+ provider |
| Fallback tự động | Cần tự viết | Có sẵn |
| Cost tracking | Qua Langfuse | Có sẵn + Langfuse |

**Quyết định:** Dùng `init_chat_model` vì project chỉ cần 3 provider (OpenAI, Google, Ollama), ưu tiên đơn giản và ít dependency. Nếu sau này cần nhiều provider hoặc fallback tự động → chuyển sang LiteLLM.

## Cấu trúc project

```
ai-assistant/
├── app/
│   ├── main.py              # FastAPI app
│   ├── api/
│   │   └── health.py        # GET /api/health
│   ├── core/
│   │   ├── config.py        # Pydantic Settings
│   │   ├── llm.py           # Lớp cấu hình LLM đa mô hình
│   │   └── tracing.py       # Langfuse integration
│   └── schemas/
│       └── health.py        # Response models
├── scripts/
│   └── compare_models.py    # So sánh 3 mô hình
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Dữ liệu demo RAG và NLQ

- `data/demo_sales_warehouse/`: 12 tài liệu tiếng Việt mẫu (MD, DOCX, PDF), từ điển dữ liệu và nguồn tham khảo.
- `data/postgres/schema_seed.sql`: tạo 9 bảng có khóa ngoại/ràng buộc và nạp 2.346 dòng dữ liệu giả lập cố định.

Tạo database rỗng rồi nạp bằng PostgreSQL client:

```bash
createdb ai_assistant_demo
psql -d ai_assistant_demo -f data/postgres/schema_seed.sql
```

Toàn bộ tên và email khách hàng trong seed là giả lập (`example.test`). Các quy chế/chính sách trong tài liệu cũng là nội dung mô phỏng để thử nghiệm, không dùng thay quy định doanh nghiệp thật.

## Benchmark mô hình

Đặt API keys trong `.env`, cài dependencies từ `requirements.txt`, bảo đảm Ollama đang chạy và model `gemma3:12b` đã được tải. Sau đó chạy từ thư mục repo:

```powershell
.venv\Scripts\python.exe -m scripts.compare_models
```

Benchmark ghi độ trễ, token, trạng thái và ước tính chi phí cho các model Groq đã cấu hình vào `docs/benchmark_runs/`. Tóm tắt lần chạy gần nhất nằm trong `docs/benchmark_results.md`.
