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
pip install -r requirements.lock

# 2. Cấu hình
cp .env.example .env
# Sửa .env: điền API key, chọn model
# Thay YOUR_USER và YOUR_PASSWORD trong DATABASE_URL bằng thông tin PostgreSQL cục bộ.
# File .env đã được Git bỏ qua; không commit thông tin đăng nhập thật.

# 3. Chạy
python -m app.main

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
| Local inference | `LOCAL_MODEL` | `ollama/gemma3:12b` |
| Mô hình trả lời RAG | `RAG_MODEL` | `local` |

Groq có thể dùng với `GROQ_API_KEY` trong `.env`, ví dụ `groq/openai/gpt-oss-120b` và `groq/openai/gpt-oss-20b`. Gemini vẫn được hỗ trợ bằng `GOOGLE_API_KEY` và model có tiền tố `google_genai/`.

RAG dùng role local theo mặc định (`RAG_MODEL=local`); có thể đổi sang role hoặc tên model khác trong `.env`. Local inference không gửi prompt tới nhà cung cấp model cloud. Tuy nhiên, nếu Langfuse Cloud được cấu hình, callback tracing có thể gửi câu hỏi, ngữ cảnh được truy xuất và câu trả lời lên Langfuse; không bật tracing cloud nếu dữ liệu phải ở hoàn toàn trên máy.

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
│   │   ├── health.py        # Health endpoints
│   │   └── rag.py           # POST /api/rag/ask
│   ├── core/
│   │   ├── config.py        # Pydantic Settings
│   │   ├── llm.py           # Lớp cấu hình LLM đa mô hình
│   │   ├── ingestion.py     # Đọc và chia chunk tài liệu
│   │   ├── tracing.py       # Langfuse integration
│   │   └── vector_store.py  # PostgreSQL/pgvector
│   └── schemas/
│       ├── health.py        # Health response
│       └── rag.py           # RAG request/response
├── data/
│   ├── demo_sales_warehouse/ # Tài liệu và SQL seed mẫu
│   ├── documents/            # Corpus chính sách thay thế (không nạp mặc định)
│   ├── rag_golden_set.json
│   └── nlq_golden_set.json
├── docs/
│   ├── architecture.md
│   ├── benchmark_results.md
│   └── rag_evaluation_v1.md
├── eval/
│   └── report_v1.md
├── scripts/
│   ├── compare_models.py
│   ├── ingest_documents.py
│   ├── evaluate_rag_retrieval.py
│   ├── evaluate_rag_answers.py
│   ├── evaluate_nlq_golden.py
│   └── evaluate_nlq_model.py
├── tests/
│   ├── test_evaluators.py
│   ├── test_health.py
│   ├── test_ingestion.py
│   ├── test_llm.py
│   ├── test_rag.py
│   ├── test_vector_store.py
│   └── test_tracing.py
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── requirements.lock
└── README.md
```

## Dữ liệu demo RAG và NLQ

- `data/demo_sales_warehouse/`: 12 tài liệu tiếng Việt mẫu (MD, DOCX, PDF), từ điển dữ liệu và nguồn tham khảo.
- `data/demo_sales_warehouse/schema_seed.sql`: tạo 9 bảng có khóa ngoại/ràng buộc và nạp 2.346 dòng dữ liệu giả lập cố định.

Tạo database rỗng rồi nạp bằng PostgreSQL client:

```bash
createdb ai_assistant_demo
psql -d ai_assistant_demo -f data/demo_sales_warehouse/schema_seed.sql
```

Toàn bộ tên và email khách hàng trong seed là giả lập (`example.test`). Các quy chế/chính sách trong tài liệu cũng là nội dung mô phỏng để thử nghiệm, không dùng thay quy định doanh nghiệp thật.

### Nạp tài liệu và hỏi đáp RAG

Mặc định corpus gồm 12 tài liệu nghiệp vụ trong `data/demo_sales_warehouse`; ba file mô tả repo/dataset (`README.md`, `data_dictionary.md`, `nguon_dataset.md`) được loại khỏi ingestion. `data/documents/` là corpus chính sách riêng, không được trộn vào corpus bán hàng mặc định; muốn nạp corpus này thì đổi `RAG_DOCUMENT_DIR=data/documents` rồi chạy ingest lại. Đảm bảo PostgreSQL có extension pgvector, Ollama đang chạy, sau đó tải model embedding và cấu hình `DATABASE_URL`, `RAG_DOCUMENT_DIR` trong `.env`:

```bash
ollama pull bge-m3
python -m scripts.ingest_documents
```

Pipeline đọc MD/TXT/DOCX/PDF, chia chunk rồi ghi embedding vào bảng `rag_chunks` (tự tạo nếu chưa có). `RAG_MIN_RELEVANCE_SCORE` cấu hình ngưỡng để API từ chối câu hỏi không có đoạn liên quan đủ mạnh. Hỏi tài liệu qua:

```bash
curl -X POST http://localhost:8000/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Nhân viên được nghỉ phép năm bao nhiêu ngày?","top_k":5}'
```

Phản hồi gồm `answer` và danh sách `sources` có đường dẫn, đoạn, điểm và số trang (với PDF). API lọc từng chunk dưới ngưỡng và trả `sources: []` khi từ chối trả lời. Chạy `python -m scripts.evaluate_rag_retrieval` để đo Hit@1/3/5, MRR và độ bao phủ nguồn yêu cầu theo nhóm câu; multi-document chỉ được tính hit khi tìm thấy đủ tất cả nguồn. `python -m scripts.evaluate_rag_answers` kiểm tra chuỗi đáp án trên API đang chạy, gồm từng mảnh đáp án với câu tổng hợp. Đây là bộ đo phát triển trên corpus nhỏ, không phải đánh giá held-out. NLQ golden script chỉ xác nhận SQL tham chiếu khớp dữ liệu seed, không đo khả năng sinh SQL của mô hình.

Chạy `python -m scripts.evaluate_nlq_golden` để kiểm tra SQL tham chiếu với seed; chạy `python -m scripts.evaluate_nlq_model --model-role local` để chấm SQL do model sinh bằng kết quả thực thi read-only. Lệnh sau gọi model cho từng câu trong bộ golden.

`requirements.lock` khóa phiên bản runtime và dev đã resolve cho nhiều nền tảng. Khi thay dependencies, cập nhật bằng `uv pip compile requirements-dev.txt --output-file requirements.lock --universal`, review diff rồi cài lại từ lock.

API demo không có xác thực hay giới hạn tần suất. Mặc định chỉ bind vào `127.0.0.1`; trước khi mở ra mạng hoặc dùng thật, cần bổ sung xác thực và rate limiting.

## Benchmark mô hình

Đặt API keys trong `.env`, cài dependencies từ `requirements.txt`, bảo đảm Ollama đang chạy và model `gemma3:12b` đã được tải. Sau đó chạy từ thư mục repo:

```powershell
.venv\Scripts\python.exe -m scripts.compare_models
```

Benchmark ghi độ trễ, token, trạng thái và ước tính chi phí cho các model Groq đã cấu hình vào `docs/benchmark_runs/`. Tóm tắt lần chạy gần nhất nằm trong `docs/benchmark_results.md`.
