# Kiến trúc hệ thống Mini AI Assistant

> **Phiên bản**: 0.2.0
> **Cập nhật**: 2026-10-26
> **Tác giả**: Trần Quốc Triệu – Intern AI, IOC

## 1. Tổng quan

Mini AI Assistant hiện có API hỏi đáp RAG và các script đánh giá NLQ:

- **RAG (Retrieval-Augmented Generation)**: Tìm kiếm và trả lời câu hỏi từ tài liệu nội bộ tiếng Việt (PDF/DOCX/MD/TXT).
- **NLQ (Natural Language Query)**: Chuyển câu hỏi tự nhiên thành truy vấn SQL trên cơ sở dữ liệu PostgreSQL.

API được xây dựng trên FastAPI + LangChain, hỗ trợ đa mô hình (local & cloud), và tích hợp Langfuse tracing tùy cấu hình. NLQ có endpoint runtime dựa trên LangGraph; script evaluator offline vẫn được dùng để so sánh khả năng sinh SQL trên golden set.

## 2. Sơ đồ kiến trúc tổng thể

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI["REST Client"]
    end

    subgraph API["FastAPI Backend"]
        GW["API Gateway\n/api/rag/ask\n/api/nlq/ask\n/api/health"]
        MW["Middleware\nCORS · Auth · Logging"]
    end

    subgraph RAG_Pipeline["RAG Pipeline"]
        RL["Document Loader\nLangChain TextLoader / PyPDFLoader\npython-docx for ordered DOCX structure"]
        SP["Section-aware chunking\nparagraph-packed, 900 / 150"]
        EM["Embedding\nbge-m3"]
        VS["Vector Store\nPGVector"]
        RET["Retriever\ncosine similarity"]
        SYN["LCEL answer chain\n(Gemini by default)"]
    end

    subgraph NLQ_Pipeline["NLQ Pipeline"]
        SC["Cached schema loader\nPostgreSQL catalog"]
        SG["LangGraph\nlexical scope + SQL generation"]
        EX["SQL guard + read-only executor\nPostgreSQL"]
    end

    subgraph LLM_Layer["Multi-Model LLM Layer"]
        LM["get_llm(role)\nProvider Routing"]
        OL["Ollama\ngemma3:12b\n(local)"]
        OA["OpenAI\n(optional provider)"]
        GG["Google GenAI\nGemini 3.5 Flash-Lite / 3.6 Flash\n(default cloud models)"]
    end

    subgraph Observability["Observability"]
        LF["Langfuse Cloud\nTracing · Eval · Dashboard"]
    end

    subgraph Data["Data Layer"]
        PG["PostgreSQL\nBusiness DB\n9 bảng · 2346 rows"]
        PGV["PGVector\nDocument Embeddings"]
        DOCS["Tài liệu nội bộ\n12 files\nMD · DOCX · PDF"]
    end

    UI --> GW
    GW --> MW --> RET --> SYN
    RL --> SP --> EM --> VS
    VS --> RET
    SC --> SG --> EX

    SYN --> LM
    SG --> LM

    LM --> OL
    LM --> OA
    LM --> GG

    VS --> PGV
    EX --> PG
    RL --> DOCS

    GW -.->|"callbacks"| LF
    LM -.->|"traces"| LF
    GW --> SC

    style Client fill:#e3f2fd,stroke:#1565c0
    style API fill:#fff3e0,stroke:#ef6c00
    style RAG_Pipeline fill:#e8f5e9,stroke:#2e7d32
    style NLQ_Pipeline fill:#fce4ec,stroke:#c62828
    style LLM_Layer fill:#fff8e1,stroke:#f9a825
    style Observability fill:#e0f7fa,stroke:#00838f
    style Data fill:#f5f5f5,stroke:#616161
```

## 3. Mô tả các component

### 3.1 API Gateway (FastAPI)

| Endpoint | Mô tả | Trạng thái |
|----------|--------|:----------:|
| `GET /api/health` | Readiness check (PostgreSQL, NLQ DB, Ollama; Langfuse status) | ✅ |
| `GET /api/health/live` | Liveness probe | ✅ |
| `GET /api/health/verify-model` | Kiểm tra model (debug only, cần API key) | ✅ |
| `POST /api/ask` | Truy vấn hợp nhất (Auto/RAG/NLQ) kèm 1-step fallback an toàn | ✅ |
| `POST /api/rag/ask` | Hỏi đáp từ tài liệu (JSON response) | ✅ |
| `POST /api/rag/ask/stream` | Hỏi đáp tài liệu dạng streaming Server-Sent Events (SSE) | ✅ |
| `POST /api/rag/upload` | Tải lên tài liệu và phân vùng corpus theo session (tối đa 5MB) | ✅ |
| `POST /api/nlq/ask` | Hỏi dữ liệu PostgreSQL bằng ngôn ngữ tự nhiên, chỉ đọc | ✅ |
| `POST /api/feedback` | Ghi nhận phản hồi người dùng (thumbs up/down) gắn trace trên Langfuse | ✅ |

**Lifespan**: Khởi tạo tracing → kiểm tra Ollama async → flush traces khi shutdown.

Các endpoint RAG và kiểm tra model yêu cầu `X-API-Key`; `/api/health` và `/api/health/live` công khai. API được giới hạn mặc định 60 request/phút theo IP bằng bộ đếm trong tiến trình. Cấu hình nhiều worker/reverse proxy cần giới hạn ở proxy hoặc shared store.

### 3.2 Multi-Model LLM (`app/core/llm.py`)

```
┌─────────────────────────────────────────────────┐
│  get_llm(role, temperature, timeout, **kwargs)  │
├─────────────────────────────────────────────────┤
│  1. _resolve_model_name(role)                   │
│     "strong" → Gemini 3.6 Flash                  │
│     "fast"   → Gemini 3.5 Flash-Lite             │
│     "local"  → "ollama/gemma3:12b"              │
│  2. _parse_provider_and_model(model_str)        │
│     "ollama/gemma3:12b" → ("ollama","gemma3:12b")│
│  3. Provider-specific kwargs:                    │
│     Ollama  → client_kwargs.timeout              │
│     Cloud   → timeout, max_retries              │
│  4. init_chat_model(provider, model)            │
│  5. Cache theo (model, temp, timeout, kwargs)   │
└─────────────────────────────────────────────────┘
```

**Chiến lược model theo role**:

| Role | Mục đích | Model mặc định | Ghi chú |
|------|----------|-----------------|---------|
| `local` | Dữ liệu nhạy cảm, offline | `ollama/gemma3:12b` | ~6 tok/s trên CPU |
| `fast` | RAG, NLQ, rewrite | `google_genai/gemini-3.5-flash-lite` | Cần `GOOGLE_API_KEY` |
| `strong` | SQL phức tạp, synthesize | `google_genai/gemini-3.6-flash` | Cần `GOOGLE_API_KEY` |

### 3.3 Request flow, Tracing & Streaming

```mermaid
flowchart TD
    REQ["POST /api/rag/ask hoặc /api/rag/ask/stream"] --> SESS["Session Store\nLấy lịch sử hội thoại"]
    SESS --> REW["Question Rewriter\nViết lại câu hỏi tỉnh lược (nếu có lịch sử)"]
    REW --> GATE["retrieve_and_gate()\nVector / Hybrid + Cổng Cosine Gate (>= 0.5)"]
    GATE -->|Từ chối| NO_ANS["Trả về NO_ANSWER\nLưu turn vào session"]
    GATE -->|Đủ bằng chứng| RERANK["Optional Reranker\nbge-reranker-v2-m3 (top-n)"]
    RERANK --> PROMPT["Langfuse Prompt Management\n(Fallback hardcoded prompt)"]
    PROMPT --> LLM["LLM Answer Generation\n(Local Gemma3:12b / Cloud)"]
    LLM --> RESP["JSON / SSE Streaming\nLưu turn vào session"]
```

`/api/rag/ask` hỗ trợ trả về JSON có cấu trúc (`RagAskResponse`), trong khi `/api/rag/ask/stream` hỗ trợ Server-Sent Events (SSE) trả về từng token (`text/event-stream`). Cả 2 endpoint đều đi qua cổng ngữ cảnh (`session_store`), viết lại câu hỏi phụ thuộc ngữ cảnh (`question_rewriter`), và cổng retrieval (`retrieve_and_gate`). Khi điểm tương đồng cosine cao nhất dưới ngưỡng (`0.5`), hệ thống lập tức từ chối mà không gọi LLM suy luận, bảo đảm tiết kiệm tài nguyên và phòng tránh hallucination.

### 3.4 RAG Pipeline & Retrieval Abstraction

Pipeline retrieval có một đường thực thi tại `app/core/retriever.py`:

1. **Vector**: truy vấn cosine trên PGVector bằng `bge-m3`.
2. **Hybrid**: trộn danh sách BM25 và vector bằng weighted Reciprocal Rank Fusion (hiện dùng trọng số đều 0.5/0.5). Token BM25 chuẩn hóa Unicode/chữ hoa-thường và tách bỏ dấu câu; mỗi chunk được định danh bằng cặp `(source, chunk_index)` để không gộp các đoạn trùng chữ từ hai file. Chỉ mục BM25 kiểm tra revision DB tối đa mỗi 5 giây; ingest trong cùng tiến trình chủ động xóa cache ngay.
3. **Cosine gate**: chấm cosine cho từng chunk sau fusion và chỉ đưa chunk vượt `RAG_MIN_RELEVANCE_SCORE` vào context. Truy vấn hybrid tái sử dụng cùng một embedding câu hỏi cho cả truy vấn vector và chấm các chunk BM25-only.
4. **Reranker**: khi bật, lấy tối đa 20 ứng viên (hoặc `top_k` nếu lớn hơn), cosine-gate rồi xếp hạng lại và trả `top_k`. Hàm `rerank_documents` dùng `sentence-transformers` với model lấy từ `RERANKER_MODEL`; khởi tạo model có lock để tránh tải trùng đồng thời. Cài tùy chọn nặng bằng `pip install -r requirements-reranker.txt`; startup kiểm tra dependency khi tính năng được bật.
5. **Langfuse Prompt Management**: tải prompt dạng chat, chuyển danh sách message thành `ChatPromptTemplate`, dùng client SDK dùng chung để tái sử dụng prompt cache và truyền fallback hardcoded cho lần tải đầu. Các script đánh giá dùng prompt đã chốt trong repo và ghi nguồn prompt trong artifact.

Lịch sử hội thoại hiện lưu trong bộ nhớ của một tiến trình, giữ tối đa 20 lượt gần nhất mỗi phiên, xóa phiên không hoạt động 30 phút và giới hạn 1.000 phiên theo LRU. Dùng một worker; nhiều worker hoặc khởi động lại sẽ không chia sẻ/giữ lịch sử, nên cần PostgreSQL/Redis store nếu triển khai nhiều tiến trình.


### 3.5 NLQ API và evaluator

```mermaid
flowchart LR
    A["POST /api/nlq/ask"] --> B["LangGraph\nlexical scope check"]
    B --> C["Generate and validate SQL"]
    C --> D["Read-only PostgreSQL\nstatement timeout + row cap"]
    D --> E["Interpret rows\n+ truncation flag"]
    G["Offline golden-set evaluator"] --> H["Seed DDL + configured model"]
    H --> I["Compare SQL result rows"]
```

`POST /api/nlq/ask` runs the production LangGraph pipeline with lexical scope detection, the cached full schema catalog (10-minute TTL), SQL validation, pooled read-only execution, and Vietnamese answer interpretation. Scope detection is local and does not call an LLM. An in-scope request uses one SQL-generation call and one interpretation call, plus at most two repairs; interpretation receives at most 25 result rows. The offline evaluator uses the same SQL extraction and validation functions, but supplies schema from seed DDL, so its scores still do not measure the production schema-loading path.

### 3.6 Langfuse Observability (`app/core/tracing.py`)

```mermaid
flowchart LR
    REQ["Request"] --> CTX["langfuse_context()\nsession_id, user_id, tags"]
    CTX --> HANDLER["CallbackHandler\n(per-request)"]
    HANDLER --> TRACE["Trace\nLangfuse Cloud"]
    TRACE --> DASH["Dashboard\nLatency · Token · Cost"]
    TRACE --> EVAL["Evaluation\nCustom Langfuse evaluators"]

    SHUT["App Shutdown"] --> FLUSH["flush_langfuse()"]
    FLUSH --> TRACE
```

**Thiết kế an toàn**:
- `is_langfuse_configured()` kiểm tra key trước khi tạo handler → app vẫn chạy khi thiếu key
- `init_tracing()` gọi tường minh trong lifespan, không phụ thuộc thứ tự import
- `langfuse_context()` chỉ bọc phần init trong try, yield ngoài → không che lỗi gốc

### 3.7 Data Layer

#### Tài liệu nội bộ (12 files)
- **Chủ đề**: Quy chế công ty giả lập "Sao Việt" (làm việc, bảo mật, lương, đổi trả, bảo hành, vận chuyển, khuyến mãi, nhập/xuất kho, đơn hàng, FAQ)
- **Định dạng**: 6 Markdown + 3 DOCX + 3 PDF
- **Vị trí**: `data/demo_sales_warehouse/`

#### Cơ sở dữ liệu PostgreSQL

```mermaid
erDiagram
    categories ||--o{ products : phan_loai
    suppliers ||--o{ products : cung_cap
    customers ||--o{ orders : dat
    orders ||--|{ order_items : gom
    products ||--o{ order_items : duoc_ban
    warehouses ||--o{ inventory : chua
    products ||--o{ inventory : ton
    warehouses ||--o{ stock_movements : phat_sinh
    products ||--o{ stock_movements : bien_dong
```

| Bảng | Số dòng | Mô tả |
|------|---------|-------|
| categories | 8 | Danh mục sản phẩm |
| suppliers | 15 | Nhà cung cấp |
| products | 80 | Sản phẩm (SKU, giá, trạng thái) |
| customers | 200 | Khách hàng (hạng thành viên, tỉnh/thành) |
| warehouses | 3 | Kho hàng |
| orders | 300 | Đơn hàng (trạng thái, phí giao) |
| order_items | 900 | Chi tiết đơn (số lượng, giá) |
| inventory | 240 | Tồn kho hiện tại |
| stock_movements | 600 | Biến động kho (nhập/xuất/điều chỉnh) |
| **Tổng** | **~2.346** | |

## 4. Công nghệ sử dụng

| Layer | Công nghệ | Phiên bản |
|-------|-----------|-----------|
| Backend | FastAPI + Uvicorn | 0.115+ |
| Orchestration | Direct FastAPI flow + LangChain LCEL & retrievers | 1.4.7+ |
| LLM Local | Ollama (gemma3:12b) | latest |
| LLM Cloud | Google GenAI (default), OpenAI (optional) | Gemini 3.5 Flash-Lite, Gemini 3.6 Flash |
| Embedding | bge-m3 (BAAI) | 1024-dim |
| Hybrid Retrieval | rank-bm25 + PGVector (EnsembleRetriever RRF) | 0.2.2 |
| Reranker | BAAI/bge-reranker-v2-m3 (sentence-transformers) | 6.2.0 |
| Vector DB | PostgreSQL + pgvector | 16+ |
| Tracing & Prompts | Langfuse Cloud | v5+ |
| Testing | pytest + pytest-asyncio | 9.1+ |
| Frontend (dự kiến) | Streamlit | 1.38+ |

## 5. Cấu trúc thư mục

```
ai-assistant/
├── app/
│   ├── main.py              # FastAPI entrypoint, lifespan
│   ├── api/
│   │   ├── health.py        # Health check endpoints
│   │   └── rag.py           # POST /api/rag/ask & /api/rag/ask/stream
│   ├── core/
│   │   ├── config.py        # Pydantic Settings (.env)
│   │   ├── llm.py           # Multi-model LLM factory
│   │   ├── retriever.py     # Unified retrieval (Vector, Hybrid, Gate)
│   │   ├── reranker.py      # Cross-encoder reranker
│   │   ├── session_store.py # Multi-turn conversational memory
│   │   ├── question_rewriter.py # Context-aware question rewriter
│   │   ├── rag_prompt.py    # Langfuse Prompt Management & hardcoded prompt
│   │   ├── security.py      # API key validation & rate limiting
│   │   ├── tracing.py       # Langfuse integration
│   │   └── vector_store.py  # PGVector database client
│   └── schemas/             # Pydantic request/response models
├── data/
│   ├── demo_sales_warehouse/ # 12 tài liệu nghiệp vụ
│   ├── conversation_dev_set.json # 10 kịch bản test hội thoại
│   ├── rag_golden_set.json  # 65 test cases đánh giá RAG
│   └── nlq_golden_set.json  # 30 test cases NLQ
├── eval/
│   ├── baseline_v5_65cases.json
│   ├── hybrid_vs_vector_comparison.md
│   ├── hybrid_vs_vector_comparison.json
│   ├── rerank_config_comparison.md
│   ├── rerank_config_comparison.json
│   └── conversation_dev_evaluation.md
├── docs/
│   ├── architecture.md      # (file này)
│   ├── benchmark_results.md # Kết quả benchmark LLM
│   └── rag_evaluation_v1.md
├── scripts/
│   ├── evaluate_hybrid_vs_vector.py # Thực nghiệm AI09
│   ├── evaluate_rerank_configs.py   # Thực nghiệm AI10
│   ├── evaluate_chat_history.py     # Thực nghiệm AI11
│   ├── validate_golden_data.py
│   └── ingest_documents.py
├── tests/                   # 115 unit & integration tests
├── .env.example
├── requirements.lock
├── requirements-reranker.txt # Sentence-transformers/Torch (optional)
└── README.md
```

## 6. Luồng xử lý hiện tại

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI
    participant RET as PGVector retrieval
    participant LLM as LCEL answer chain
    participant LF as Langfuse

    User->>API: POST /api/rag/ask
    API->>LF: request observation + session context
    API->>RET: similarity_search(question, top_k)
    RET-->>API: chunks, scores, source metadata
    API->>LF: retrieval span (when configured)
    API->>LLM: ainvoke(question, context, callbacks)
    LLM-->>API: answer text
    API->>API: attach sources from retrieved chunks
    API-->>User: JSON response
```

## 7. Quyết định thiết kế quan trọng

1. **Workflow hiện tại**
   - RAG là luồng trực tiếp trong FastAPI; LCEL chain kết hợp `ChatPromptTemplate`, chat model và Pydantic output parser.
   - NLQ runtime API dùng LangGraph, lexical scope check và read-only SQL; evaluator dùng seed DDL để so sánh model.

2. **Tại sao multi-model?**
   - Local model cho dữ liệu nhạy cảm và offline
   - Cloud model cho chất lượng và tốc độ
   - Fallback: `local.with_fallbacks([cloud])` với timeout

3. **Tại sao Langfuse thay vì LangSmith?**
   - Hỗ trợ self-host (tương lai)
   - Tích hợp tốt với LangChain qua CallbackHandler
   - Dashboard trace; evaluator hiện tại là custom LLM-as-a-judge, không phải Ragas.

4. **Tại sao PostgreSQL + pgvector?**
   - Một database cho cả business data (NLQ) và vector store (RAG)
   - Giảm complexity vận hành so với dùng Chroma/Pinecone riêng
   - pgvector hỗ trợ HNSW index cho production

## 8. Benchmark sơ bộ (gemma3:12b local)

| Prompt Type | Thời gian | Tokens | Tốc độ |
|-------------|-----------|--------|--------|
| RAG (trả lời ngắn) | 22.19s | ~133 | ~6 tok/s |
| SQL Generation | 81.83s | ~491 | ~6 tok/s |
| Reasoning (dài) | 263.20s | ~1491 | ~5.7 tok/s |

> **Nhận xét**: Tốc độ đều ~6 tok/s cho thấy nhiều khả năng chạy CPU. Cần kiểm tra `ollama ps` và cân nhắc bản quantized nhỏ hơn cho tác vụ nhẹ.

## 9. Rủi ro và giải pháp

| Rủi ro | Giải pháp |
|--------|-----------|
| Local model chậm (CPU-bound) | Streaming SSE, fallback sang cloud, giới hạn max_tokens |
| SQL injection từ NLQ | Chỉ cho phép SELECT, validate trước khi execute, LIMIT mặc định |
| Concurrent requests lẫn trace | Handler per-request, propagate_attributes dùng contextvars |
| Tài liệu tiếng Việt chunk sai | Tùy chỉnh text splitter cho tiếng Việt, test với bộ golden set |
| PostgreSQL chưa cài | Dùng Docker Compose hoặc cài native cho dev |


## 10. Ước tính chi phí vận hành trên 1.000 câu hỏi (Cost Estimation)

Giả định phân bổ lưu lượng thực tế trong doanh nghiệp (Mixed Enterprise Traffic):
- **50% RAG (500 câu)**: Hỏi đáp chính sách, bảo hành, quy định nội bộ (~2.150 input tokens, ~300 output tokens / câu).
- **35% NLQ (350 câu)**: Truy vấn số liệu bán hàng, đơn hàng, khách hàng (~1.650 input tokens, ~230 output tokens / câu).
- **10% Chào hỏi / Ngoài phạm vi (100 câu)**: (~150 input tokens, ~30 output tokens / câu).
- **5% Thao tác cấm / Vi phạm chính sách (50 câu)**: Chặn ở router & regex policy guard: **0 LLM token**.

**Tổng lượng Token ước tính cho 1.000 câu hỏi:**
- **Input Tokens**: `(500 * 2.150) + (350 * 1.650) + (100 * 150) ≈ 1.667.500` tokens (~1.67M tokens).
- **Output Tokens**: `(500 * 300) + (350 * 230) + (100 * 30) ≈ 233.500` tokens (~0.23M tokens).

| Cấu hình Mô hình | Đơn giá Input (trên 1M) | Đơn giá Output (trên 1M) | Chi phí Input (1.67M) | Chi phí Output (0.23M) | **Tổng Chi Phí / 1.000 câu** | Quy đổi VNĐ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phương án 1: Google Gemini 3.5 Flash-Lite / 2.0 Flash (Khuyến nghị Production)** | $0.075 | $0.30 | $0.125 | $0.069 | **$0.194** | **~4.900 VNĐ** |
| **Phương án 2: OpenAI GPT-4o-mini** | $0.150 | $0.60 | $0.251 | $0.138 | **$0.389** | **~9.800 VNĐ** |
| **Phương án 3: Google Gemini 3.6 Flash** | $0.350 | $1.05 | $0.585 | $0.242 | **$0.827** | **~20.700 VNĐ** |
| **Phương án 4: Anthropic Claude 3.5 Haiku** | $0.800 | $4.00 | $1.336 | $0.920 | **$2.256** | **~56.400 VNĐ** |
| **Phương án 5: Local 100% (Ollama Gemma3:12b + BGE-M3)** | $0.000 | $0.000 | $0.000 | $0.000 | **$0.000** | **0 VNĐ** *(On-premise)* |

> [!TIP]
> **Khuyến nghị tối ưu chi phí & hiệu năng**:
> - Dùng **Gemini 3.5 Flash-Lite** làm cấu hình mặc định (Production): Chỉ tốn **chưa tới 5.000 VNĐ cho mỗi 1.000 lượt hỏi**, tốc độ phản hồi cực nhanh (TTFT < 800ms).
> - Với các dữ liệu nhạy cảm cao hoặc triển khai nội bộ hoàn toàn (On-Premise / Air-gapped): Sử dụng **Ollama (Gemma3:12b / Qwen 2.5) + BGE-M3**, chi phí API bằng **0 đồng**.
