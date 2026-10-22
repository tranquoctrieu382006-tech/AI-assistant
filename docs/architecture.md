# Kiến trúc hệ thống Mini AI Assistant

> **Phiên bản**: 0.1.0
> **Cập nhật**: 2026-10-22
> **Tác giả**: Trần Quốc Triệu – Intern AI, IOC

## 1. Tổng quan

Mini AI Assistant là hệ thống trợ lý AI nội bộ tích hợp hai chức năng chính:

- **RAG (Retrieval-Augmented Generation)**: Tìm kiếm và trả lời câu hỏi từ tài liệu nội bộ tiếng Việt (PDF/DOCX/MD).
- **NLQ (Natural Language Query)**: Chuyển câu hỏi tự nhiên thành truy vấn SQL trên cơ sở dữ liệu PostgreSQL.

Hệ thống được xây dựng trên FastAPI + LangChain + LangGraph, hỗ trợ đa mô hình (local & cloud), và tích hợp Langfuse để theo dõi chất lượng.

## 2. Sơ đồ kiến trúc tổng thể

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI["Streamlit UI / REST Client"]
    end

    subgraph API["FastAPI Backend"]
        GW["API Gateway\n/api/rag/ask\n/api/nlq/ask\n/api/health"]
        MW["Middleware\nCORS · Auth · Logging"]
    end

    subgraph Orchestration["LangGraph Orchestration"]
        RT["Intent Router\n(fast model)"]
        RAG["RAG Subgraph"]
        NLQ["NLQ Subgraph"]
    end

    subgraph RAG_Pipeline["RAG Pipeline"]
        RL["Document Loader\nDocling / LangChain"]
        SP["Text Splitter\nRecursiveCharacter"]
        EM["Embedding\nbge-m3"]
        VS["Vector Store\nPGVector"]
        RET["Retriever\nMMR / Similarity"]
        SYN["Synthesizer\n(strong model)"]
    end

    subgraph NLQ_Pipeline["NLQ Pipeline"]
        SC["Schema Loader\nData Dictionary"]
        SG["SQL Generator\n(strong model)"]
        EX["SQL Executor\nPostgreSQL"]
        FMT["Answer Formatter\n(fast model)"]
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
    GW --> MW --> RT

    RT -->|"intent=rag"| RAG
    RT -->|"intent=nlq"| NLQ

    RAG --> RET --> SYN
    RL --> SP --> EM --> VS

    NLQ --> SC --> SG --> EX --> FMT

    SYN --> LM
    SG --> LM
    FMT --> LM
    RT --> LM

    LM --> OL
    LM --> OA
    LM --> GG

    RET --> PGV
    VS --> PGV
    EX --> PG
    RL --> DOCS

    GW -.->|"callbacks"| LF
    LM -.->|"traces"| LF

    style Client fill:#e3f2fd,stroke:#1565c0
    style API fill:#fff3e0,stroke:#ef6c00
    style Orchestration fill:#f3e5f5,stroke:#7b1fa2
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
| `GET /api/health/verify-model` | Kiểm tra model (debug only) | ✅ |
| `POST /api/rag/ask` | Hỏi đáp từ tài liệu | 🔜 Tuần 2 |
| `POST /api/nlq/ask` | Hỏi đáp từ database | 🔜 Tuần 2 |

**Lifespan**: Khởi tạo tracing → kiểm tra Ollama async → flush traces khi shutdown.

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

### 3.3 LangGraph Orchestration (Tuần 2)

```mermaid
stateDiagram-v2
    [*] --> Router
    Router --> RAG: intent = rag
    Router --> NLQ: intent = nlq
    Router --> Clarify: intent = unclear

    RAG --> Rewrite: query cần cải thiện
    Rewrite --> RAG
    RAG --> Synthesize
    Synthesize --> [*]

    NLQ --> GenerateSQL
    GenerateSQL --> ValidateSQL
    ValidateSQL --> ExecuteSQL: SQL hợp lệ
    ValidateSQL --> GenerateSQL: SQL lỗi (retry)
    ExecuteSQL --> FormatAnswer
    FormatAnswer --> [*]

    Clarify --> [*]
```

**Nguyên tắc gắn tracing vào LangGraph**:
- Handler gắn ở cấp `graph.invoke(..., config={"callbacks": [handler]})`, không gắn từng LLM
- Mỗi request tạo handler riêng (tránh lẫn session khi concurrent)
- Đặt tên node và tag theo role (`router`, `sql_gen`, `synthesize`)

### 3.4 RAG Pipeline

```mermaid
flowchart LR
    A["Tài liệu\n(PDF/DOCX/MD)"] --> B["Document Loader\n(Docling/LangChain)"]
    B --> C["Text Splitter\nchunk_size=1000\noverlap=200"]
    C --> D["Embedding\nbge-m3"]
    D --> E["PGVector\nPostgreSQL"]
    E --> F["Retriever\nMMR k=5"]
    F --> G["Prompt + Context"]
    G --> H["LLM\n(strong model)"]
    H --> I["Câu trả lời\n+ nguồn trích dẫn"]
```

### 3.5 NLQ Pipeline

```mermaid
flowchart LR
    A["Câu hỏi\ntiếng Việt"] --> B["Schema Context\nData Dictionary"]
    B --> C["SQL Generator\n(strong model)"]
    C --> D["SQL Validator\nsyntax + safety"]
    D --> E["PostgreSQL\nExecutor"]
    E --> F["Result Formatter\n(fast model)"]
    F --> G["Câu trả lời\n+ SQL đã chạy"]
```

**An toàn SQL**: Chỉ cho phép SELECT, cấm DROP/DELETE/UPDATE/INSERT. Giới hạn LIMIT mặc định.

### 3.6 Langfuse Observability (`app/core/tracing.py`)

```mermaid
flowchart LR
    REQ["Request"] --> CTX["langfuse_context()\nsession_id, user_id, tags"]
    CTX --> HANDLER["CallbackHandler\n(per-request)"]
    HANDLER --> TRACE["Trace\nLangfuse Cloud"]
    TRACE --> DASH["Dashboard\nLatency · Token · Cost"]
    TRACE --> EVAL["Evaluation\nRagas · Custom"]

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
| Orchestration | LangChain + LangGraph | 0.3+ |
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

## 6. Luồng xử lý request (dự kiến)

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI
    participant LG as LangGraph
    participant Router as Intent Router
    participant RAG as RAG Chain
    participant NLQ as NLQ Chain
    participant LLM as LLM (get_llm)
    participant DB as PostgreSQL
    participant LF as Langfuse

    User->>API: POST /api/rag/ask hoặc /api/nlq/ask
    API->>LF: langfuse_context(session_id, user_id)
    API->>LG: graph.invoke(query, config={callbacks})

    LG->>Router: classify intent
    Router->>LLM: get_llm("fast")

    alt RAG
        LG->>RAG: retrieve + synthesize
        RAG->>DB: PGVector similarity search
        DB-->>RAG: relevant chunks
        RAG->>LLM: get_llm("strong") + context
        LLM-->>RAG: answer
    else NLQ
        LG->>NLQ: generate SQL + execute
        NLQ->>LLM: get_llm("strong") + schema
        LLM-->>NLQ: SQL query
        NLQ->>DB: execute SELECT
        DB-->>NLQ: result rows
        NLQ->>LLM: get_llm("fast") + format
        LLM-->>NLQ: formatted answer
    end

    LG-->>API: response
    API->>LF: flush traces (at shutdown)
    API-->>User: JSON response
```

## 7. Quyết định thiết kế quan trọng

1. **Tại sao LangGraph thay vì chain đơn giản?**
   - Cần router giữa RAG và NLQ
   - NLQ cần retry loop khi SQL lỗi
   - RAG có thể cần rewrite query
   - Dễ mở rộng thêm tác vụ mới

2. **Tại sao multi-model?**
   - Local model cho dữ liệu nhạy cảm và offline
   - Cloud model cho chất lượng và tốc độ
   - Fallback: `local.with_fallbacks([cloud])` với timeout

3. **Tại sao Langfuse thay vì LangSmith?**
   - Hỗ trợ self-host (tương lai)
   - Tích hợp tốt với LangChain qua CallbackHandler
   - Dashboard đánh giá chất lượng (Ragas integration)

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
