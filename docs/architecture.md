# Kiến trúc hệ thống Mini AI Assistant

> **Phiên bản**: 0.1.0
> **Cập nhật**: 2026-10-23
> **Tác giả**: Trần Quốc Triệu – Intern AI, IOC

## 1. Tổng quan

Mini AI Assistant hiện có API hỏi đáp RAG và các script đánh giá NLQ:

- **RAG (Retrieval-Augmented Generation)**: Tìm kiếm và trả lời câu hỏi từ tài liệu nội bộ tiếng Việt (PDF/DOCX/MD/TXT).
- **NLQ (Natural Language Query)**: Chuyển câu hỏi tự nhiên thành truy vấn SQL trên cơ sở dữ liệu PostgreSQL.

API được xây dựng trên FastAPI + LangChain, hỗ trợ đa mô hình (local & cloud), và tích hợp Langfuse tracing tùy cấu hình. NLQ hiện là evaluator offline; chưa có endpoint NLQ trong API. LangGraph chưa được dùng trong luồng runtime hiện tại.

## 2. Sơ đồ kiến trúc tổng thể

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI["REST Client"]
    end

    subgraph API["FastAPI Backend"]
        GW["API Gateway\n/api/rag/ask\n/api/health"]
        MW["Middleware\nCORS · Auth · Logging"]
    end

    subgraph RAG_Pipeline["RAG Pipeline"]
        RL["Document Loader\npython-docx / pypdf / text"]
        SP["Section-aware chunking\nparagraph-packed, 900 / 150"]
        EM["Embedding\nbge-m3"]
        VS["Vector Store\nPGVector"]
        RET["Retriever\ncosine similarity"]
        SYN["LCEL answer chain\n(local by default)"]
    end

    subgraph NLQ_Pipeline["NLQ Pipeline"]
        SC["Schema Loader\nseed DDL"]
        SG["SQL Generator\nevaluation script"]
        EX["SQL Executor\nPostgreSQL"]
    end

    subgraph LLM_Layer["Multi-Model LLM Layer"]
        LM["get_llm(role)\nProvider Routing"]
        OL["Ollama\ngemma3:12b\n(local)"]
        OA["OpenAI\ngpt-4o / gpt-4o-mini\n(cloud)"]
        GG["Google GenAI\ngemini-2.0-flash\n(cloud)"]
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
| `GET /api/health` | Readiness check (Ollama, Langfuse) | ✅ |
| `GET /api/health/live` | Liveness probe | ✅ |
| `GET /api/health/verify-model` | Kiểm tra model (debug only, cần API key) | ✅ |
| `POST /api/rag/ask` | Hỏi đáp từ tài liệu | ✅ |

**Lifespan**: Khởi tạo tracing → kiểm tra Ollama async → flush traces khi shutdown.

Các endpoint RAG và kiểm tra model yêu cầu `X-API-Key`; `/api/health` và `/api/health/live` công khai. API được giới hạn mặc định 60 request/phút theo IP bằng bộ đếm trong tiến trình. Cấu hình nhiều worker/reverse proxy cần giới hạn ở proxy hoặc shared store.

### 3.2 Multi-Model LLM (`app/core/llm.py`)

```
┌─────────────────────────────────────────────────┐
│  get_llm(role, temperature, timeout, **kwargs)  │
├─────────────────────────────────────────────────┤
│  1. _resolve_model_name(role)                   │
│     "strong" → "openai/gpt-4o"                  │
│     "fast"   → "openai/gpt-4o-mini"             │
│     "local"  → "ollama/gemma3:12b"              │
│  2. _parse_provider_and_model(model_str)        │
│     "ollama/gemma3:12b" → ("ollama","gemma3:12b")│
│  3. Provider-specific kwargs:                    │
│     Ollama  → client_kwargs.timeout              │
│     OpenAI  → timeout, max_retries              │
│  4. init_chat_model(provider, model)            │
│  5. Cache theo (model, temp, timeout, kwargs)   │
└─────────────────────────────────────────────────┘
```

**Chiến lược model theo role**:

| Role | Mục đích | Model mặc định | Ghi chú |
|------|----------|-----------------|---------|
| `local` | Dữ liệu nhạy cảm, offline | `ollama/gemma3:12b` | ~6 tok/s trên CPU |
| `fast` | Router, rewrite, format | `openai/gpt-4o-mini` | Phản hồi nhanh, chi phí thấp |
| `strong` | SQL phức tạp, synthesize | `openai/gpt-4o` | Chất lượng cao nhất |

### 3.3 Request flow and tracing

```mermaid
flowchart LR
    REQ["POST /api/rag/ask"] --> ROOT["Request observation"]
    ROOT --> RET["Retrieval span"]
    RET --> CTX["Filtered context"]
    CTX --> LLM["LCEL\nChatPromptTemplate | ChatModel | StrOutputParser"]
    LLM --> RESP["Answer + retrieval sources"]
```

`/api/rag/ask` dựng LCEL chain `ChatPromptTemplate | ChatModel | StrOutputParser`; đầu ra API vẫn là chuỗi answer, còn `sources` được tạo từ kết quả retrieval thay vì citation do LLM phát ra. Request observation bao ngoài retrieval span và callback được truyền vào chain. LangGraph chưa được dùng.

Chunking mặc định gộp các paragraph trong cùng section/page cho đến giới hạn kích thước. `paragraph_blocks` là biến thể đánh giá giữ riêng ranh giới block được trích xuất. So sánh trên golden set ngày 2026-10-23 không cho thấy lợi ích của biến thể này (Hit@3 0.974 so với 1.000); mặc định hiện tại không thay đổi. Xem kết quả trong `eval/chunking_comparison_2026-10-23.json`.

Một lời gọi HTTP trực tiếp tới `/api/rag/ask` với retriever/model và dữ liệu tổng hợp đã được xác nhận xuất hiện trong Langfuse Cloud, gồm các observation request, retrieval và LCEL/model. Phép thử xác nhận đường truyền tracing; nó không kiểm tra provider model thật hoặc dữ liệu retrieval thật.

### 3.4 RAG Pipeline

```mermaid
flowchart LR
    A["Tài liệu\n(PDF/DOCX/MD/TXT)"] --> B["python-docx / pypdf / text"]
    B --> C["Chunker\n900 chars / 150 overlap"]
    C --> D["Embedding\nbge-m3"]
    D --> E["PGVector\nPostgreSQL"]
    E --> F["Retriever\ncosine similarity k=5"]
    F --> G["Prompt + Context"]
    G --> H["LLM\n(configured RAG model; local by default)"]
    H --> I["Câu trả lời\n+ nguồn từ retrieval"]
```

Markdown headings and DOCX Heading styles are stored as section metadata; PDF chunks retain page numbers. Other formats or PDF headings are not inferred. Re-run document ingestion after deployment to apply the section-column migration and rebuild chunks.

### 3.5 NLQ model evaluator (offline)

```mermaid
flowchart LR
    A["Câu hỏi golden\n(tiếng Việt)"] --> B["Schema Context\nseed DDL"]
    B --> C["SQL Generator\n(configured model)"]
    C --> D["SQLGlot + read-only guard"]
    D --> E["PostgreSQL\nExecutor"]
    E --> F["Compare with expected rows"]
```

This is an offline evaluation script, not an API pipeline. It accepts one SELECT/WITH statement, executes it in a read-only transaction with a timeout, and compares rows with the golden set. There is no NLQ endpoint or answer formatter in the current app.

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
| Orchestration | Direct FastAPI flow + LangChain model calls | Current |
| LLM Local | Ollama (gemma3:12b) | latest |
| LLM Cloud | OpenAI API, Google GenAI | gpt-4o, gemini-2.0-flash |
| Embedding | bge-m3 (BAAI) | – |
| Vector DB | PostgreSQL + pgvector | 16+ |
| Tracing | Langfuse Cloud | v5+ |
| Testing | pytest + pytest-asyncio | 9.1+ |
| Frontend (dự kiến) | Streamlit | 1.38+ |

## 5. Cấu trúc thư mục

```
ai-assistant/
├── app/
│   ├── main.py              # FastAPI entrypoint, lifespan
│   ├── api/
│   │   └── health.py        # Health check endpoints
│   ├── core/
│   │   ├── config.py        # Pydantic Settings (.env)
│   │   ├── llm.py           # Multi-model LLM factory
│   │   └── tracing.py       # Langfuse integration
│   └── schemas/             # Pydantic request/response models
├── data/
│   ├── demo_sales_warehouse/
│   │   ├── 01-12_*.md|docx|pdf  # 12 tài liệu tiếng Việt
│   │   ├── schema_seed.sql       # 9 bảng, ~2346 rows
│   │   └── data_dictionary.md    # Từ điển dữ liệu + ERD
│   └── documents/                # Bộ tài liệu phụ (7 files MD)
├── docs/
│   ├── architecture.md           # (file này)
│   ├── benchmark_results.md      # Kết quả benchmark LLM
│   └── benchmark_runs/           # CSV exports
├── scripts/
│   └── compare_models.py         # Script benchmark
├── tests/
│   ├── test_health.py            # 4 tests (mocked)
│   ├── test_llm.py               # 13 tests
│   └── test_tracing.py           # 7 tests
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
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
   - RAG là luồng trực tiếp trong FastAPI; LCEL chain kết hợp `ChatPromptTemplate`, chat model và `StrOutputParser`.
   - NLQ model evaluation là script offline dùng seed DDL và read-only SQL.
   - LangGraph và NLQ API chưa được triển khai.

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
