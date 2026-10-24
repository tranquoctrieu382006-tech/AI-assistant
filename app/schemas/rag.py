from pydantic import BaseModel, Field


class RagAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=128)


class RagSource(BaseModel):
    source: str
    chunk_index: int
    score: float
    page_number: int | None = None
    section: str | None = None


class RagAskResponse(BaseModel):
    answer: str
    sources: list[RagSource]


class RagGeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1, description="Câu trả lời tiếng Việt hoặc thông báo từ chối.")
    abstain: bool = Field(
        description="True nếu context không đủ thông tin để trả lời câu hỏi."
    )
