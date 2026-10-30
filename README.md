# AI Assistant

Mini AI Assistant – Hệ thống Trợ lý Doanh nghiệp hỗ trợ tra cứu chính sách nội bộ (RAG) và truy vấn dữ liệu kinh doanh bằng ngôn ngữ tự nhiên (NLQ).

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

- **Ngôn ngữ & Framework:** Python 3.12+, FastAPI, Streamlit, LangChain, LangGraph.
- **Cơ sở dữ liệu:** PostgreSQL 16 kết hợp extension `pgvector` (Vector Store).
- **Mô hình nhúng (Embedding):** `bge-m3` (1024 chiều) chạy qua Ollama hoặc API.
- **Mô hình ngôn ngữ (LLM):** Hỗ trợ đa mô hình linh hoạt qua cấu hình môi trường:
  - `local`: `ollama/gemma3:12b` (chạy offline bảo mật dữ liệu).
  - `fast`: `google_genai/gemini-3.5-flash-lite` hoặc `openai/gpt-4o-mini` (định tuyến, viết lại câu hỏi, giải nghĩa nhanh).
  - `strong`: `google_genai/gemini-3.6-flash` hoặc `openai/gpt-4o` (tổng hợp SQL phức tạp).
- **Giám sát & Đánh giá (Observability):** Langfuse Tracing, LLM-as-a-Judge.

---

## 🏛️ Sơ Đồ Kiến Trúc Hệ Thống (Architecture Overview)

```mermaid
flowchart TD
    User["Người dùng / Web Client"] --> UI["Streamlit Chat UI (:8501)"]
    UI --> API["FastAPI Backend (:8000)"]
    API --> Orch["Unified Orchestrator (/api/ask)"]
    
    Orch --> Router{"Smart Router (Phân Loại & Bảo Vệ)"}
    Router -->|Chào hỏi xã giao| Greet["👋 Chào mừng & Hướng dẫn"]
    Router -->|Lệnh ghi phá hoại| Guard["⚠️ Chặn phá hoại (Write Guard)"]
    Router -->|Ngoài phạm vi| OOS["ℹ️ Từ chối an toàn (Abstention)"]
    Router -->|Chính sách / Tra cứu| RAG["Luồng RAG (Tra cứu tài liệu)"]
    Router -->|Số liệu / Báo cáo| NLQ["Luồng NLQ (Text-to-SQL CSDL)"]
    
    NLQ -->|PII khách hàng| PII["🔒 Chặn thông tin nhạy cảm"]
    NLQ -->|Không có cột CSDL| Fallback["🔄 Fallback 1 bước sang RAG"]
    
    RAG --> PGVector[("PostgreSQL 16 + pgvector")]
    NLQ --> PGReadOnly[("PostgreSQL ai_nlq_reader")]
    
    RAG --> LLM["Multi-LLM Provider (Gemini / GPT / Ollama)"]
    NLQ --> LLM
    
    Orch -.->|Traces & Metrics| Langfuse[("Langfuse Cloud / Self-hosted")]
    RAG -.-> Langfuse
    NLQ -.-> Langfuse
```

---

## 🚀 Khởi Động Nhanh (Quick Start)

Bạn có thể lựa chọn 1 trong 2 cách triển khai dưới đây:

### Cách 1: Sử dụng Docker Compose (Khuyến nghị cho môi trường mới)

Chỉ với 1 lệnh, hệ thống sẽ tự động khởi tạo PostgreSQL có sẵn `pgvector`, nạp 2.346 dòng dữ liệu bán hàng mẫu, tạo role chỉ đọc `ai_nlq_reader` và bật Ollama:

```bash
# 1. Khởi động hạ tầng CSDL và Ollama
docker compose up -d

# 2. Tải model embedding bge-m3 vào container Ollama
docker compose exec ollama ollama pull bge-m3

# 3. Cài đặt thư viện Python trên máy
pip install -r requirements.lock

# 4. Sao chép và cấu hình file môi trường
cp .env.example .env

# 5. Lập chỉ mục tài liệu vào PostgreSQL pgvector
python -m scripts.ingest_documents --strategy heading

# 6. Khởi động API Backend
uvicorn app.main:app --port 8000 --reload
```

---

### Cách 2: Cài đặt thủ công cục bộ (Local Setup)

```bash
# 1. Cài đặt dependencies từ lockfile
pip install -r requirements.lock

# 2. Thiết lập cấu hình .env
cp .env.example .env
# Tạo API key bí mật tối thiểu 32 ký tự:
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Điền khóa vừa sinh vào biến APP_API_KEY trong .env

# 3. Tạo CSDL PostgreSQL và nạp seed dữ liệu demo
createdb ai_assistant_demo
psql -d ai_assistant_demo -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d ai_assistant_demo -f data/demo_sales_warehouse/schema_seed.sql

# 4. Tạo role chỉ đọc an toàn cho tính năng NLQ (xem docs/threat_model_nlq.md)
psql -d ai_assistant_demo -c "
CREATE ROLE ai_nlq_reader LOGIN PASSWORD 'secret_nlq_reader';
GRANT CONNECT ON DATABASE ai_assistant_demo TO ai_nlq_reader;
GRANT USAGE ON SCHEMA public TO ai_nlq_reader;
GRANT SELECT ON categories, suppliers, products, warehouses, orders, order_items, inventory, stock_movements TO ai_nlq_reader;
GRANT SELECT (customer_id, city, member_tier, joined_on) ON customers TO ai_nlq_reader;
"

# 5. Tải model embedding và ingest tài liệu RAG
ollama pull bge-m3
python -m scripts.ingest_documents --strategy heading

# 6. Khởi chạy Backend API
uvicorn app.main:app --port 8000 --reload
```

---

## 🖥️ Giao Diện Trò Chuyện (Streamlit Chat UI)

Khởi động giao diện Web tương tác trực quan:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

**Tính năng nổi bật:**
- 🎯 **Tự động định tuyến (Smart Router):** Tự động phân loại câu hỏi sang RAG (tài liệu), NLQ (truy vấn CSDL số liệu) hoặc Out-of-scope (chào hỏi/ngoài phạm vi).
- ⚡ **Streaming SSE:** Truyền token phản hồi theo thời gian thực kèm hộp trạng thái hiển thị các bước phân tích.
- 📊 **Trực quan hóa SQL:** Xem câu lệnh SQL đã chạy, hiển thị bảng số liệu và tự động vẽ biểu đồ cột cho dữ liệu định lượng.
- 📚 **Minh bạch trích dẫn (Grounding):** Hiển thị rõ tên file nguồn, tiêu đề mục, trang PDF và điểm tương đồng cosine.
- 📁 **Nạp tài liệu tức thì:** Kéo thả tài liệu (`.pdf`, `.docx`, `.md`, `.txt`) để lập chỉ mục trực tiếp vào hệ thống.
- 👍/👎 **Phản hồi người dùng:** Đánh giá chất lượng câu trả lời, gửi điểm số trực tiếp lên Langfuse Cloud kèm `trace_id`.

---

## ⚙️ Cấu Hình Mô Hình (.env)

Thay đổi mô hình linh hoạt qua biến môi trường mà **không cần sửa code**:

| Vai trò | Biến môi trường | Giá trị khuyến nghị | Mục đích sử dụng |
| :--- | :--- | :--- | :--- |
| **Mặc định** | `DEFAULT_MODEL` | `google_genai/gemini-3.5-flash-lite` | Mô hình chính toàn hệ thống |
| **Mạnh** | `STRONG_MODEL` | `google_genai/gemini-3.6-flash` | Tổng hợp báo cáo, viết SQL phức tạp |
| **Nhanh/Rẻ** | `FAST_MODEL` | `google_genai/gemini-3.5-flash-lite` | Phân loại router, viết lại câu hỏi |
| **Nội bộ** | `LOCAL_MODEL` | `ollama/gemma3:12b` | Chạy offline, bảo mật tuyệt đối dữ liệu |
| **Tìm kiếm** | `SEARCH_MODE` | `hybrid` hoặc `vector` | Kết hợp PGVector (Dense) + BM25 (Sparse) |
| **Reranker** | `USE_RERANKER` | `false` hoặc `true` | Bật mô hình BAAI cross-encoder reranking |

---

## 📚 Hướng Dẫn Index & Re-Index Tài Liệu (RAG Ingestion)

Hệ thống hỗ trợ 2 chiến lược phân đoạn văn bản (chunking strategies):
- **`heading` (Khuyến nghị):** Phân đoạn bám sát cấu trúc tiêu đề (`#`, `##`), bảo toàn trọn vẹn ngữ cảnh điều khoản chính sách, quy chế nghiệp vụ.
- **`fixed_size`:** Phân đoạn theo kích thước cố định kèm overlap (`chunk_size=900`, `chunk_overlap=150`).

### 1. Chạy Index / Re-Index toàn bộ kho tài liệu:
```bash
# Nạp tài liệu từ data/demo_sales_warehouse theo chiến lược tiêu đề
python -m scripts.ingest_documents --strategy heading

# Hoặc nạp theo kích thước cố định
python -m scripts.ingest_documents --strategy fixed_size
```

### 2. Nạp tài liệu động trực tiếp (Runtime Upload):
- **Qua giao diện Web:** Mở sidebar Streamlit `http://localhost:8501`, kéo thả file `.pdf`, `.docx`, `.txt`, `.md` vào ô **"Tải lên tài liệu"**.
- **Qua REST API:**
  ```bash
  curl -X POST "http://localhost:8000/api/rag/upload" \
       -H "X-API-Key: YOUR_APP_API_KEY" \
       -F "file=@tai_lieu_moi.pdf"
  ```

---

## 🔄 Hướng Dẫn Đổi Mô Hình & Tùy Chỉnh Prompt

### 1. Đổi Mô Hình LLM (Không cần sửa code)
Tất cả các mô hình được quản trị tập trung qua biến môi trường trong file `.env`:
- **Đổi sang OpenAI:**
  ```env
  DEFAULT_MODEL=openai/gpt-4o-mini
  STRONG_MODEL=openai/gpt-4o
  OPENAI_API_KEY=sk-...
  ```
- **Đổi sang chạy 100% Local (Offline, bảo mật tuyệt đối):**
  ```env
  DEFAULT_MODEL=ollama/gemma3:12b
  LOCAL_MODEL=ollama/gemma3:12b
  RAG_MODEL=local
  NLQ_MODEL=local
  ```
- **Chuyển đổi linh hoạt giữa các tác vụ:**
  - `RAG_MODEL=fast` hoặc `RAG_MODEL=strong` hoặc `RAG_MODEL=local`
  - `NLQ_MODEL=strong` (để sinh SQL độ chính xác cao nhất)

### 2. Tùy Chỉnh Prompt
- **Qua Langfuse Prompt Management (Không cần restart server):**
  Hệ thống tích hợp sẵn với Langfuse Prompt Catalog (`LANGFUSE_PROMPT_NAME_RAG=rag-qa-prompt`). Khi chỉnh sửa nội dung prompt hoặc system message trên dashboard Langfuse với nhãn `production` hoặc `staging`, Backend sẽ tự động cập nhật cache theo chu kỳ TTL.
- **Chỉnh sửa Fallback Hardcoded Prompt (Offline):**
  File định nghĩa prompt mặc định: `app/core/rag_prompt.py` (cho RAG) và `app/core/nlq_graph.py` (cho NLQ Text-to-SQL).

---

## 📊 Langfuse Dashboard & Giám Sát Vận Hành

Hệ thống tích hợp sâu với Langfuse Cloud để giám sát toàn diện chất lượng:
1. **Theo dõi chi phí (Cost Tracking):** Thống kê chi tiết token tiêu thụ (Prompt tokens, Completion tokens) và chi phí USD theo từng model trên mỗi lượt gọi.
2. **Đo lường độ trễ (Latency Observability):** Theo dõi thời gian thực hiện của từng span con:
   - `router` (phân loại ý định: < 50ms)
   - `retriever` (truy xuất văn bản kết hợp: ~300ms - 800ms)
   - `nlq-sql-generation` & `nlq-sql-validation` (~1.5s - 3s)
3. **Chỉ số chất lượng thực nghiệm (Empirical Quality Metrics):**
   - **Đánh giá từ người dùng thực tế:** Điểm 👍 (1.0) / 👎 (0.0) gửi qua endpoint `POST /api/feedback` ghi nhận trực tiếp vào trace Langfuse.
   - **Đo lường RAG (Golden set 65 ca — xem chi tiết tại `eval/current_numbers.md`):**
     * **Abstention Accuracy (Từ chối an toàn khi thiếu dữ liệu):** **98.5%** (64/65 ca quyết định đúng, Hallucination trên câu hỏi âm = 0/24).
     * **Hit@5 Retrieval (Tỷ lệ tìm thấy tài liệu liên quan):** **97.6%** (40/41 ca answerable, MRR = 0.9207).
     * **Fragment Match (Khớp nội dung trả lời chuẩn):** **31–32/41 (75.6% – 78.0%)**.
   - **Đo lường NLQ (Golden set 30 ca Text-to-SQL — xem `eval/report_v2.md` §3.1):**
     * **Execution Accuracy (Khớp chính xác SQL lần đầu):** **18–20/30 (60.0% – 66.7%)** (Gemini 3.1 Flash-Lite: **20/30 = 66.7%**, Gemini 3.5 Flash-Lite: **18/30 = 60.0%**, Local Gemma3:12b: **16/30 = 53.3%**; sau cơ chế Self-Repair 2 vòng của LangGraph đạt **24/30 = 80.0%**).

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```
ai-assistant/
├── app/
│   ├── main.py                    # Khởi tạo ứng dụng FastAPI và middleware
│   ├── streamlit_app.py           # Giao diện Chat Web UI tương tác
│   ├── api/
│   │   ├── rag.py                 # POST /api/rag/ask, /stream, /upload
│   │   ├── nlq.py                 # POST /api/nlq/ask
│   │   ├── feedback.py            # POST /api/feedback (Langfuse scores)
│   │   ├── router_api.py          # POST /api/router/classify
│   │   └── health.py              # GET /api/health, /live, /ready
│   ├── core/
│   │   ├── config.py              # Pydantic Settings đọc cấu hình từ .env
│   │   ├── router.py              # Phân loại intent (RAG vs NLQ vs Out-of-scope)
│   │   ├── nlq_graph.py           # LangGraph workflow cho Text-to-SQL an toàn
│   │   ├── nlq_schema.py          # Quản lý schema & schema-linking CSDL
│   │   ├── retriever.py           # Hybrid search (Vector + BM25) & score gating
│   │   ├── question_rewriter.py   # Viết lại câu hỏi tỉnh lược qua lịch sử hội thoại
│   │   ├── rag_prompt.py          # Quản lý prompt và Langfuse prompt fallback
│   │   ├── vector_store.py        # Tương tác PostgreSQL pgvector
│   │   ├── security.py            # Xác thực API Key & Rate limiting
│   │   └── tracing.py             # Langfuse tracing & OpenTelemetry metrics
│   └── schemas/                   # Pydantic contracts cho API RAG, NLQ, Feedback
├── data/
│   ├── demo_sales_warehouse/      # 12 tài liệu nghiệp vụ mẫu & schema_seed.sql
│   ├── rag_golden_set.json        # 65 bộ câu hỏi - đáp chuẩn đánh giá RAG
│   └── nlq_golden_set.json        # 30 bộ câu hỏi - SQL chuẩn đánh giá NLQ
├── docker/
│   └── init-db/                   # Script khởi tạo extension pgvector và role DB
├── docs/
│   ├── architecture.md            # Thiết kế kiến trúc tổng thể hệ thống
│   ├── rag_api.md                 # Đặc tả chi tiết RAG API & sự kiện SSE Streaming
│   ├── nlq_api.md                 # Hướng dẫn cấu hình và sử dụng NLQ API
│   ├── threat_model_nlq.md        # Mô hình đe dọa và bảo mật phân tầng CSDL NLQ
│   └── archive/                   # Lưu trữ snapshot lịch sử (benchmark, v1 eval, worklog)
├── eval/
│   └── report_v2.md               # Báo cáo đánh giá toàn diện v2 (LLM-as-a-judge)
├── scripts/
│   ├── ingest_documents.py        # Lập chỉ mục tài liệu vào vector store
│   ├── validate_golden_data.py    # Kiểm tra tính toàn vẹn của tập Golden Dataset
│   ├── evaluate_rag_retrieval.py  # Đo lường Hit@K, MRR, Context Recall
│   └── evaluate_nlq_model.py      # Benchmark độ chính xác sinh SQL của LLM
├── tests/                         # Bộ kiểm thử tự động (Unit & Integration tests)
├── docker-compose.yml             # Triển khai nhanh PostgreSQL + pgvector + Ollama
├── Dockerfile                     # Dockerfile đóng gói ứng dụng
└── LICENSE                        # MIT License
```

---

## 🧪 Hướng Dẫn Chạy Kiểm Thử (Unit Tests)

Bộ kiểm thử được thiết kế hoàn toàn cô lập, không gọi tới API bên ngoài và không tiêu tốn token thật:

```bash
# 1. Cài đặt công cụ kiểm thử
pip install -r requirements-dev.txt

# 2. Chạy toàn bộ test suite
pytest tests/ -v

# 3. Chạy kiểm tra tính an toàn của NLQ (SQL Guard & Role Defense)
pytest tests/test_nlq_runtime.py -v

# 4. Kiểm tra tính toàn vẹn của dữ liệu mẫu (Golden Dataset)
python -m scripts.validate_golden_data
```

---

## ❓ Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Hiện tượng lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **`HTTP 503: Service Unavailable` khi gọi API** | Chưa cấu hình `APP_API_KEY` hoặc độ dài khóa ngắn hơn 32 ký tự. | Chạy lệnh `python -c "import secrets; print(secrets.token_urlsafe(32))"` và điền vào biến `APP_API_KEY` trong file `.env`. |
| **`ImportError: No module named rank_bm25`** | Bật `SEARCH_MODE=hybrid` nhưng môi trường thiếu thư viện BM25. | Cài đặt bổ sung: `pip install rank-bm25`. |
| **`psycopg.OperationalError: type "vector" does not exist`** | PostgreSQL chưa được cài đặt extension pgvector. | Chạy lệnh SQL với quyền superuser: `CREATE EXTENSION IF NOT EXISTS vector;` hoặc dùng image Docker `pgvector/pgvector:pg16`. |
| **`/api/health` báo trạng thái `degraded`** | Không kết nối được tới Ollama (cổng 11434) hoặc PostgreSQL (cổng 5432). | Kiểm tra dịch vụ Ollama (`ollama serve`) và trạng thái CSDL PostgreSQL. |
| **NLQ trả về lỗi `503: Chưa cấu hình PostgreSQL chỉ đọc`** | Biến `NLQ_DATABASE_URL` trong `.env` đang để trống. | Cấu hình chuỗi kết nối của role `ai_nlq_reader`: `NLQ_DATABASE_URL=postgresql://ai_nlq_reader:secret_nlq_reader@localhost:5432/ai_assistant_demo`. |

---

## 💰 Ước Tính Chi Phí Vận Hành (Cost per 1k Queries)

Dựa trên lưu lượng phân bổ thực tế trong doanh nghiệp (50% RAG, 35% NLQ, 10% Chào hỏi, 5% Chặn an toàn):

| Phương Án Triển Khai | Mô Hình Sử Dụng | Chi Phí / 1.000 Câu Hỏi | Ước Tính VNĐ |
| :--- | :--- | :---: | :---: |
| **Cloud Tiết Kiệm (Khuyến nghị)** | **Gemini 3.5 Flash-Lite** | **$0.19** | **~4.900 VNĐ** |
| **Cloud Tiêu Chuẩn** | **OpenAI GPT-4o-mini** | **$0.39** | **~9.800 VNĐ** |
| **Cloud Hiệu Năng Cao** | **Gemini 3.6 Flash** | **$0.83** | **~20.700 VNĐ** |
| **On-Premise (Nội bộ 100%)** | **Ollama Gemma3:12b + BGE-M3** | **$0.00** | **0 VNĐ** |

*Chi tiết cách tính toán token input/output cho từng luồng: xem tại [docs/architecture.md#10](docs/architecture.md).*

---

## 📚 Tài Liệu Tham Khảo Thêm

- [Báo Cáo Tự Kiểm Thử Tích Hợp 8 Kịch Bản (TASK-AI24)](docs/task_ai24_self_test_report.md)
- [Kiến trúc chi tiết hệ thống](docs/architecture.md)
- [Kịch bản Demo chi tiết 20-30 phút (Cheat Sheet 10 câu mẫu)](docs/demo_script_week2.md)
- [Đặc tả RAG API & Server-Sent Events (SSE)](docs/rag_api.md)
- [Hướng dẫn cấu hình NLQ API](docs/nlq_api.md)
- [Mô hình đe dọa và An toàn CSDL NLQ](docs/threat_model_nlq.md)
- [Kho lưu trữ dữ liệu lịch sử](docs/archive/)

---

## 📄 Bản Quyền (License)

Dự án được phân phối theo giấy phép [MIT License](LICENSE).
