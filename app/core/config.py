"""
Application settings – đọc từ .env, đổi model chỉ bằng biến môi trường.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cấu hình ứng dụng, tự động đọc từ file .env."""

    # ── LLM Models ──────────────────────────────────────────────
    default_model: str = "ollama/gemma3:12b"
    strong_model: str = "openai/gpt-4o"
    fast_model: str = "openai/gpt-4o-mini"
    local_model: str = "ollama/gemma3:12b"
    ollama_base_url: str = "http://localhost:11434"

    # ── API Keys ────────────────────────────────────────────────
    openai_api_key: str = ""
    google_api_key: str = ""

    # ── Langfuse ────────────────────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # ── App ─────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
