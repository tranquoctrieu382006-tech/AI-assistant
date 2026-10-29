from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
CORPUS_PATTERN = r"^(demo|user|session_[A-Za-z0-9._:-]{1,128})$"


class RagAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=128, pattern=SESSION_ID_PATTERN)
    corpus: str | None = Field(
        default=None,
        min_length=1,
        max_length=136,
        pattern=CORPUS_PATTERN,
        description="Tùy chọn lọc theo kho tài liệu: 'demo', 'user' hoặc corpus của chính session hiện tại",
    )


class RagSource(BaseModel):
    source: str
    chunk_index: int
    score: float
    page_number: int | None = None
    section: str | None = None
    display_name: str | None = None
    corpus: str | None = None


class RagAskResponse(BaseModel):
    answer: str
    sources: list[RagSource]
    trace_id: str | None = Field(default=None, description="Mã trace_id của lượt hỏi trên Langfuse (nếu bật tracing)")


class RagUploadResponse(BaseModel):
    status: str = Field(default="success")
    filename: str
    display_name: str
    chunks_indexed: int
    corpus: str = Field(default="user")


class RagGeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1, description="Câu trả lời tiếng Việt hoặc thông báo từ chối.")
    abstain: bool = Field(
        description="True nếu context không đủ thông tin để trả lời câu hỏi."
    )
