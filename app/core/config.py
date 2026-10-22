"""
Application settings – đọc từ .env, đổi model chỉ bằng biến môi trường.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ── API Keys ────────────────────────────────────────────────
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""

    # ── Langfuse ────────────────────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Server Host & Port ──────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
