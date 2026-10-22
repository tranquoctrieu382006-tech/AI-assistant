"""
Pydantic schemas cho health check endpoint.
"""

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Thông tin trạng thái hệ thống và cấu hình runtime."""

    status: str = Field(default="ok", description="Trạng thái hệ thống: ok | degraded")
    version: str = Field(default="0.1.0", description="Phiên bản API")
    default_model: str = Field(..., description="Model LLM đang làm mặc định")
    available_roles: list[str] = Field(..., description="Các vai trò model có thể dùng (strong, fast, local)")
    langfuse_enabled: bool = Field(..., description="Langfuse tracing có được kích hoạt hay không")
    ollama_status: dict[str, Any] = Field(default_factory=dict, description="Trạng thái kết nối tới Ollama")
