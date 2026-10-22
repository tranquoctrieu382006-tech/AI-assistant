# AI Assistant

Mini AI Assistant â€“ Há»i Ä‘Ã¡p tÃ i liá»‡u (RAG) & truy váº¥n dá»¯ liá»‡u báº±ng ngÃ´n ngá»¯ tá»± nhiÃªn (NLQ).

## Tech Stack

- **Python 3.11+**, **FastAPI**, **LangChain + LangGraph**
- **Langfuse** (tracing, prompt management, eval)
- **PostgreSQL + pgvector** (vector store)
- **Ollama** (local LLM)

## Quick Start

```bash
# 1. Clone & cÃ i Ä‘áº·t
git clone <repo-url>
cd ai-assistant
pip install -r requirements.txt

# 2. Cáº¥u hÃ¬nh
cp .env.example .env
# Sá»­a .env: Ä‘iá»n API key, chá»n model

# 3. Cháº¡y
uvicorn app.main:app --reload

# 4. Má»Ÿ Swagger UI
# http://localhost:8000/docs
```

## Äá»•i mÃ´ hÃ¬nh

Sá»­a biáº¿n trong `.env`, **khÃ´ng cáº§n sá»­a code**:

| Vai trÃ² | Biáº¿n mÃ´i trÆ°á»ng | VÃ­ dá»¥ |
|---|---|---|
| Máº·c Ä‘á»‹nh | `DEFAULT_MODEL` | `openai/gpt-4o-mini` |
| Máº¡nh (SQL phá»©c táº¡p, tá»•ng há»£p) | `STRONG_MODEL` | `openai/gpt-4o` |
| Nhanh/ráº» (router, viáº¿t láº¡i cÃ¢u há»i) | `FAST_MODEL` | `openai/gpt-4o-mini` |
| Local (khÃ´ng gá»­i dá»¯ liá»‡u ra ngoÃ i) | `LOCAL_MODEL` | `ollama/gemma3:12b` |

Groq cÃ³ thá»ƒ dÃ¹ng vá»›i `GROQ_API_KEY` trong `.env`, vÃ­ dá»¥ `groq/openai/gpt-oss-120b` vÃ  `groq/openai/gpt-oss-20b`. Gemini váº«n Ä‘Æ°á»£c há»— trá»£ báº±ng `GOOGLE_API_KEY` vÃ  model cÃ³ tiá»n tá»‘ `google_genai/`.

## Táº¡i sao chá»n `init_chat_model` thay vÃ¬ LiteLLM?

| TiÃªu chÃ­ | init_chat_model (LangChain) | LiteLLM |
|---|---|---|
| Dependency | KhÃ´ng thÃªm package | ThÃªm `litellm` |
| TÆ°Æ¡ng thÃ­ch LangChain | Native, luÃ´n Ä‘á»“ng bá»™ version | ÄÃ´i khi lag version |
| Sá»‘ provider há»— trá»£ | ~10 provider chÃ­nh | 100+ provider |
| Fallback tá»± Ä‘á»™ng | Cáº§n tá»± viáº¿t | CÃ³ sáºµn |
| Cost tracking | Qua Langfuse | CÃ³ sáºµn + Langfuse |

**Quyáº¿t Ä‘á»‹nh:** DÃ¹ng `init_chat_model` vÃ¬ project chá»‰ cáº§n 3 provider (OpenAI, Google, Ollama), Æ°u tiÃªn Ä‘Æ¡n giáº£n vÃ  Ã­t dependency. Náº¿u sau nÃ y cáº§n nhiá»u provider hoáº·c fallback tá»± Ä‘á»™ng â†’ chuyá»ƒn sang LiteLLM.

## Cáº¥u trÃºc project

```
ai-assistant/
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ main.py              # FastAPI app
â”‚   â”œâ”€â”€ api/
â”‚   â”‚   â””â”€â”€ health.py        # GET /api/health
â”‚   â”œâ”€â”€ core/
â”‚   â”‚   â”œâ”€â”€ config.py        # Pydantic Settings
â”‚   â”‚   â”œâ”€â”€ llm.py           # Lá»›p cáº¥u hÃ¬nh LLM Ä‘a mÃ´ hÃ¬nh
â”‚   â”‚   â””â”€â”€ tracing.py       # Langfuse integration
â”‚   â””â”€â”€ schemas/
â”‚       â””â”€â”€ health.py        # Response models
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ compare_models.py    # So sÃ¡nh 3 mÃ´ hÃ¬nh
â”œâ”€â”€ .env.example
â”œâ”€â”€ .gitignore
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

## Dá»¯ liá»‡u demo RAG vÃ  NLQ

- `data/demo_sales_warehouse/`: 12 tÃ i liá»‡u tiáº¿ng Viá»‡t máº«u (MD, DOCX, PDF), tá»« Ä‘iá»ƒn dá»¯ liá»‡u vÃ  nguá»“n tham kháº£o.
- `data/demo_sales_warehouse/schema_seed.sql`: táº¡o 9 báº£ng cÃ³ khÃ³a ngoáº¡i/rÃ ng buá»™c vÃ  náº¡p 2.346 dÃ²ng dá»¯ liá»‡u giáº£ láº­p cá»‘ Ä‘á»‹nh.

Táº¡o database rá»—ng rá»“i náº¡p báº±ng PostgreSQL client:

```bash
createdb ai_assistant_demo
psql -d ai_assistant_demo -f data/demo_sales_warehouse/schema_seed.sql
```

ToÃ n bá»™ tÃªn vÃ  email khÃ¡ch hÃ ng trong seed lÃ  giáº£ láº­p (`example.test`). CÃ¡c quy cháº¿/chÃ­nh sÃ¡ch trong tÃ i liá»‡u cÅ©ng lÃ  ná»™i dung mÃ´ phá»ng Ä‘á»ƒ thá»­ nghiá»‡m, khÃ´ng dÃ¹ng thay quy Ä‘á»‹nh doanh nghiá»‡p tháº­t.

## Benchmark mÃ´ hÃ¬nh

Äáº·t API keys trong `.env`, cÃ i dependencies tá»« `requirements.txt`, báº£o Ä‘áº£m Ollama Ä‘ang cháº¡y vÃ  model `gemma3:12b` Ä‘Ã£ Ä‘Æ°á»£c táº£i. Sau Ä‘Ã³ cháº¡y tá»« thÆ° má»¥c repo:

```powershell
.venv\Scripts\python.exe -m scripts.compare_models
```

Benchmark ghi Ä‘á»™ trá»…, token, tráº¡ng thÃ¡i vÃ  Æ°á»›c tÃ­nh chi phÃ­ cho cÃ¡c model Groq Ä‘Ã£ cáº¥u hÃ¬nh vÃ o `docs/benchmark_runs/`. TÃ³m táº¯t láº§n cháº¡y gáº§n nháº¥t náº±m trong `docs/benchmark_results.md`.
