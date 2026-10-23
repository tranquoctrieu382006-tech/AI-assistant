from pydantic import BaseModel, Field


class RagAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None


class RagSource(BaseModel):
    source: str
    chunk_index: int
    score: float


class RagAskResponse(BaseModel):
    answer: str
    sources: list[RagSource]
