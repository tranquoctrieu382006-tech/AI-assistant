# AI Assistant

Mini AI Assistant – Hỏi đáp tài liệu (RAG) & truy vấn dữ liệu bằng ngôn ngữ tự nhiên (NLQ).

## Tech Stack

- **Python 3.11+**, **FastAPI**, **LangChain** (RAG API; NLQ is an offline evaluator)
- **Langfuse** (optional tracing and RAG model experiments)
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
# Tạo API key: python -c "import secrets; print(secrets.token_urlsafe(32))"
# Điền kết quả vào APP_API_KEY trong .env (không chia sẻ/commit khóa này)
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
│   │   ├── rag_prompt.py    # Prompt dùng chung cho API và evaluation
│   │   ├── security.py     # API key và rate limiting
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
│   ├── compare_chunking_strategies.py
│   ├── evaluate_rag_answers.py
│   ├── evaluate_rag_experiment.py
│   ├── evaluate_rag_structured_local.py
│   ├── evaluate_nlq_golden.py
│   └── evaluate_nlq_model.py
├── tests/
│   ├── test_evaluators.py
│   ├── test_health.py
│   ├── test_ingestion.py
│   ├── test_llm.py
│   ├── test_rag.py
│   ├── test_security.py
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
python -m scripts.ingest_documents --strategy heading
```

Pipeline mặc định chia theo heading (`--strategy heading`); có thể chọn `--strategy fixed_size` để chia kích thước cố định có overlap. Hai cách được so sánh độc lập, không ghi vào vector store, bằng `python -m scripts.compare_chunking_strategies`. Pipeline dùng LangChain `TextLoader` cho MD/TXT và `PyPDFLoader` cho PDF; DOCX tiếp tục dùng parser `python-docx` để giữ thứ tự đoạn/bảng và style Heading mà `Docx2txtLoader` làm mất. Sau đó, pipeline ghi embedding vào bảng `rag_chunks` (tự tạo nếu chưa có). Chunk giữ metadata nguồn, mục heading (Markdown/DOCX) và trang PDF; chunk fixed-size trộn nhiều mục sẽ để section trống thay vì gắn sai mục. Sau khi cập nhật phiên bản có metadata mục, cần chạy lại ingestion để migrate schema và tái lập chỉ mục. `RAG_MIN_RELEVANCE_SCORE` cấu hình ngưỡng để API từ chối câu hỏi không có đoạn liên quan đủ mạnh. Hỏi tài liệu qua:

```bash
curl -X POST http://localhost:8000/api/rag/ask \
  -H "X-API-Key: <APP_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"question":"Nhân viên được nghỉ phép năm bao nhiêu ngày?","top_k":5}'
```

Phản hồi gồm `answer` và danh sách `sources` có đường dẫn, mục, đoạn, điểm và số trang (với PDF). API lọc từng chunk dưới ngưỡng và trả `sources: []` khi từ chối trả lời. Chạy `python -m scripts.validate_golden_data` để kiểm tra mã hóa, nhãn, nguồn và evidence offline; chạy `python -m scripts.evaluate_rag_retrieval` để đo context recall@5, Hit@1/3/5, MRR và lỗi từ chối theo nhóm. Evidence được gắn bằng đoạn văn bản nguồn thay vì chunk ID; câu multi-document chỉ được tính hit khi tìm thấy đủ tất cả nguồn. `python -m scripts.evaluate_rag_answers` kiểm tra chuỗi đáp án trên API đang chạy, gồm từng mảnh đáp án với câu tổng hợp. Đây là bộ đo phát triển, không phải đánh giá held-out. Quy tắc đóng băng và tạo held-out nằm trong `docs/rag_eval_protocol.md`. NLQ golden script chỉ xác nhận SQL tham chiếu khớp dữ liệu seed, không đo khả năng sinh SQL của mô hình.

So sánh chunking hiện tại (gộp paragraph trong cùng section/page) với phương án giữ riêng block trích xuất bằng `python -m scripts.compare_chunking_strategies`. Script dùng embedding local, cùng golden set và ngưỡng retrieval; không sửa vector store đang chạy. Kết quả lần gần nhất được lưu ở `eval/chunking_comparison_2026-10-23.json`.

Chạy `python -m scripts.evaluate_nlq_golden` để kiểm tra SQL tham chiếu với seed; chạy `python -m scripts.evaluate_nlq_model --model-role local --output eval/nlq_model_local_2026-10-23.json` để chấm SQL do model sinh bằng kết quả thực thi read-only. Lệnh này gọi model cho từng câu trong bộ golden và lưu kết quả cục bộ. Sau khi đã duyệt phạm vi chia sẻ, có thể tạo Langfuse Dataset/Experiment từ kết quả đã lưu (không gọi model lại) bằng `python -m scripts.evaluate_nlq_model --upload-results eval/nlq_model_local_2026-10-23.json --langfuse --output eval/nlq_model_local_2026-10-23.json`; nếu không có `--langfuse`, dữ liệu không được gửi lên Cloud.

`python -m scripts.evaluate_rag_experiment --roles local strong --judge-role fast` so sánh câu trả lời local Gemma với model strong đã cấu hình, chấm answer substring, faithfulness và answer relevancy, rồi tạo Dataset/Experiment và bổ sung các item còn thiếu trên Langfuse. Lệnh này gửi câu hỏi, context được truy xuất, đáp án mong đợi và câu trả lời model tới nhà cung cấp model judge/candidate và Langfuse Cloud; chỉ chạy khi đã phê duyệt phạm vi chia sẻ dữ liệu. Bộ dữ liệu demo hiện là dữ liệu tổng hợp.

`requirements.lock` khóa phiên bản runtime và dev đã resolve cho nhiều nền tảng. Khi thay dependencies, cập nhật bằng `uv pip compile requirements-dev.txt --output-file requirements.lock --universal`, review diff rồi cài lại từ lock.

API `/api/rag/*` và endpoint debug `/api/health/verify-model` yêu cầu header `X-API-Key`; đặt `APP_API_KEY` thành chuỗi bí mật ngẫu nhiên tối thiểu 32 ký tự. Khi chưa cấu hình key, các endpoint được bảo vệ trả HTTP 503 thay vì mở công khai. `/api/health` và `/api/health/live` vẫn công khai để health checks. Các endpoint API được giới hạn mặc định 60 request/phút theo IP; bộ đếm nằm trong tiến trình, nên nếu chạy nhiều worker hoặc sau reverse proxy thì cần giới hạn ở proxy/chia sẻ store. Middleware không tin `X-Forwarded-For`, tránh giả mạo IP; cấu hình proxy tin cậy riêng nếu cần.

## Benchmark mô hình

Đặt API keys trong `.env`, cài dependencies từ `requirements.txt`, bảo đảm Ollama đang chạy và model `gemma3:12b` đã được tải. Sau đó chạy từ thư mục repo:

```powershell
.venv\Scripts\python.exe -m scripts.compare_models
```

Benchmark ghi độ trễ, token, trạng thái và ước tính chi phí cho các model Groq đã cấu hình vào `docs/benchmark_runs/`. Tóm tắt lần chạy gần nhất nằm trong `docs/benchmark_results.md`.
