from typing import Literal
from pydantic import BaseModel, Field


class NlqAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)


class NlqAskResponse(BaseModel):
    sql: str
    columns: list[str]
    rows: list[dict]
    answer: str
    repaired: bool = False
    truncated: bool = False
    trace_id: str | None = None
    status: Literal["ok", "restricted", "out_of_scope", "unsupported_action", "error"] = "ok"
    error_code: str | None = None
    fallback_from: str | None = None
