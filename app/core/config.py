"""
Application settings – đọc từ .env, đổi model chỉ bằng biến môi trường.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Cấu hình ứng dụng, tự động đọc từ file .env."""

    # ── App Metadata & Environment ──────────────────────────────
    app_name: str = "AI Assistant"
    app_version: str = "0.1.0"
    environment: str = "development"
    # Mặc định an toàn: debug tắt trên production, bật qua biến DEBUG=true trong .env ở dev
    debug: bool = False
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8501",  # Streamlit UI
    ]

    # ── LLM Models ──────────────────────────────────────────────
    default_model: str = "google_genai/gemini-3.5-flash-lite"
    strong_model: str = "google_genai/gemini-3.6-flash"
    fast_model: str = "google_genai/gemini-3.5-flash-lite"
    local_model: str = "ollama/gemma3:12b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_dimensions: int = Field(default=1024, ge=1)
    rag_num_ctx: int = Field(default=8192, ge=512)
    # Require an explicit local DATABASE_URL instead of silently guessing credentials.
    database_url: str = ""
    # NLQ uses an isolated read-only credential and never falls back to DATABASE_URL.
    nlq_database_url: str = ""
    nlq_model: str = "fast"
    nlq_model_timeout_seconds: float = Field(default=60.0, gt=0)
    nlq_model_max_retries: int = Field(default=1, ge=0, le=2)
    nlq_max_rows: int = Field(default=200, ge=1, le=1000)
    nlq_statement_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    nlq_max_repairs: int = Field(default=2, ge=0, le=2)
    nlq_schema_dictionary_path: str = "data/demo_sales_warehouse/data_dictionary.md"
    database_connection_timeout_seconds: float = Field(default=3.0, gt=0)
    rag_document_dir: str = "data/demo_sales_warehouse"
    rag_exclude_files: list[str] = ["README.md", "data_dictionary.md", "nguon_dataset.md"]
    # Chat, RAG, NLQ, and question rewriting default to Gemini; embeddings stay local in Ollama.
    rag_model: str = "fast"
    question_rewriter_model: str = "fast"
    rag_model_timeout_seconds: float = Field(default=30.0, gt=0)
    rag_local_model_timeout_seconds: float = Field(default=90.0, gt=0)
    rag_model_max_retries: int = Field(default=1, ge=0, le=1)
    question_rewriter_timeout_seconds: float = Field(default=10.0, gt=0)
    rag_stream_first_token_timeout_seconds: float = Field(default=35.0, gt=0)
    rag_stream_idle_timeout_seconds: float = Field(default=15.0, gt=0)
    rag_retrieval_timeout_seconds: float = Field(default=60.0, gt=0)
    bm25_revision_ttl_seconds: float = Field(default=5.0, ge=0, le=60)
    app_workers: int = Field(default=1, ge=1, le=32)
    search_mode: Literal["vector", "hybrid"] = "vector"
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_pool_size: int = Field(default=20, ge=1, le=200)
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    # Development cutoff; the corrected golden set still has three negative
    # false positives. Do not treat it as a validated production threshold.
    rag_min_relevance_score: float = 0.5
    rag_embedding_batch_size: int = Field(default=64, ge=1)
    llm_timeout_seconds: float = Field(default=90.0, gt=0)

    # ── API Keys ────────────────────────────────────────────────
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""

    # ── Langfuse ────────────────────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Server Host & Port ──────────────────────────────────────
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_api_key: str = ""
    api_rate_limit_per_minute: int = Field(default=60, ge=1)
    trusted_proxy_ips: list[str] = []

    @field_validator("rag_document_dir", "nlq_schema_dictionary_path", mode="before")
    @classmethod
    def resolve_project_paths(cls, value):
        path = Path(value)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
