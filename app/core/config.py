"""
Application settings – đọc từ .env, đổi model chỉ bằng biến môi trường.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
    default_model: str = "ollama/gemma3:12b"
    strong_model: str = "openai/gpt-4o"
    fast_model: str = "openai/gpt-4o-mini"
    local_model: str = "ollama/gemma3:12b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    # Require an explicit local DATABASE_URL instead of silently guessing credentials.
    database_url: str = ""
    rag_document_dir: str = "data/demo_sales_warehouse"
    rag_exclude_files: list[str] = ["README.md", "data_dictionary.md", "nguon_dataset.md"]
    # RAG defaults to a local model; cloud use must be an explicit choice.
    rag_model: str = "local"
    search_mode: Literal["vector", "hybrid"] = "vector"
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    # Development cutoff; the corrected golden set still has three negative
    # false positives. Do not treat it as a validated production threshold.
    rag_min_relevance_score: float = 0.5
    rag_embedding_batch_size: int = Field(default=64, ge=1)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
