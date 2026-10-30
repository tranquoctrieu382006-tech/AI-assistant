"""Context-aware question rewriter for multi-turn RAG conversations and diacritic restoration."""
from __future__ import annotations

import logging
import re
import unicodedata
import asyncio
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.core.llm import get_rag_llm
from app.core.session_store import trim_history
from app.core.text_utils import is_probably_unaccented_vietnamese, strip_accents

logger = logging.getLogger(__name__)

# Strong cues are unambiguous follow-ups and may trigger a rewrite freely.
# "thế nào" is a question phrase, not a follow-up reference, so exclude it.
_STRONG_CUES = re.compile(
    r"\b(?:"
    r"còn|thế(?!\s*nào)|vậy|nữa|trường\s+hợp\s+ấy|"
    r"đó|này|nó|họ|ấy|kia|chúng|chúng\s+nó|anh\s+ấy|chị\s+ấy|bạn\s+ấy"
    r")\b",
    re.IGNORECASE,
)
# Unaccented variants only apply to pure-ASCII questions.
_STRONG_CUES_ASCII = re.compile(
    r"\b(?:the(?!\s*nao)|vay|nua|truong\s+hop\s+ay)\b",
    re.IGNORECASE,
)
# Ambiguous cues also occur in ordinary prose ("hiện nay", "dùng chung",
# "do lỗi"). They only count for short questions or fixed collocations.
_AMBIGUOUS_CUES = re.compile(r"\b(?:do|nay|chung|trên|con)\b", re.IGNORECASE)
_AMBIGUOUS_CUES_ASCII = re.compile(
    r"\b(?:do|no|ho|con|nay|chung|tren|ay)\b", re.IGNORECASE
)
_AMBIGUOUS_COLLOCATION = re.compile(
    r"\b(?:do|no|ho)\s+(?:đó|vậy|the|vay)\b|\b(?:đó|này|kia|ấy)\s+(?:là|thì)\b",
    re.IGNORECASE,
)
_SHORT_QUESTION_WORDS = 5

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of one rewrite call.

    `accent_restored` is true only when an unaccented Vietnamese question came
    back with diacritics. `coref_resolved` is true when the rewrite changed
    anything beyond diacritics/punctuation/case, i.e. it added context from the
    conversation history.
    """

    text: str
    accent_restored: bool = False
    coref_resolved: bool = False
    failed: bool = False


def _normalize_for_compare(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace, then strip accents."""
    lowered = unicodedata.normalize("NFKC", text or "").casefold()
    lowered = _PUNCTUATION_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", strip_accents(lowered)).strip()


def _has_followup_cue(question: str) -> bool:
    """Return True when the question plausibly refers back to the history."""
    text = (question or "").strip()
    if not text:
        return False
    if _STRONG_CUES.search(text):
        return True
    if text.isascii() and _STRONG_CUES_ASCII.search(text):
        return True

    ambiguous = _AMBIGUOUS_CUES.search(text)
    if ambiguous is None and text.isascii():
        ambiguous = _AMBIGUOUS_CUES_ASCII.search(text)
    if ambiguous is None:
        return False
    if len(text.split()) < _SHORT_QUESTION_WORDS:
        return True
    return bool(_AMBIGUOUS_COLLOCATION.search(text))


REWRITE_SYSTEM_PROMPT = """Bạn là chuyên viên làm rõ và chuẩn hóa câu hỏi trong hệ thống hỏi đáp tài liệu nội bộ.
Dựa vào LỊCH SỬ HỘI THOẠI và CÂU HỎI MỚI, hãy diễn giải câu hỏi mới thành một câu hỏi ĐỘC LẬP, RÕ RÀNG bằng tiếng Việt CÓ DẤU.

Quy tắc bắt buộc:
1. Nếu câu hỏi mới dùng đại từ thay thế hoặc bị tỉnh lược (ví dụ: "thế còn hàng lỗi?", "phải gửi trước bao lâu?", "ai ký?"): bổ sung ngữ cảnh từ lượt trước để câu hỏi có thể hiểu được khi đứng một mình.
2. Nếu câu hỏi được viết bằng tiếng Việt KHÔNG DẤU (ví dụ: "quy trinh tra hang nhu the nao", "thoi han doi tra la bao lau"): hãy KHÔI PHỤC ĐẦY ĐỦ VÀ CHÍNH XÁC DẤU TIẾNG VIỆT theo ngữ cảnh văn bản nghiệp vụ. Giữ nguyên các mã định danh viết tắt (như mã đơn, mã SKU) và thuật ngữ tiếng Anh.
3. Nếu câu hỏi mới đã rõ ràng, có dấu đầy đủ và không phụ thuộc ngữ cảnh: GIỮ NGUYÊN câu hỏi.
4. CHỈ xuất ra duy nhất câu hỏi đã làm rõ/khôi phục dấu. KHÔNG giải thích, KHÔNG chào hỏi, KHÔNG trả lời câu hỏi.
"""


def build_rewriter_chain(model_role: str | None = None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM_PROMPT),
        ("human", "LỊCH SỬ HỘI THOẠI:\n{history}\n\nCÂU HỎI MỚI: {question}\n\nCÂU HỎI ĐỘC LẬP:"),
    ])
    llm = get_rag_llm(
        model_role,
        temperature=0.0,
        timeout=settings.question_rewriter_timeout_seconds,
        max_retries=0,
    )
    return prompt | llm | StrOutputParser()


async def rewrite_question(
    question: str,
    history: list[dict],
    model_role: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> RewriteResult:
    """Rewrite a question and report why it changed.

    The rewrite is skipped when the question is already accented and does not
    look like a follow-up. Any failure returns the original question with
    `failed=True` and both flags false, which callers treat as the safe path
    (no dual retrieval).
    """
    needs_accent_restoration = is_probably_unaccented_vietnamese(question)
    trimmed = trim_history(history, max_turns=5) if history else []

    needs_context_rewrite = bool(
        trimmed and (len(question.split()) < 4 or _has_followup_cue(question))
    )

    if not needs_accent_restoration and not needs_context_rewrite:
        return RewriteResult(text=question)

    history_lines = []
    for msg in trimmed:
        role_label = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        history_lines.append(f"{role_label}: {msg['content']}")
    history_text = "\n".join(history_lines) if history_lines else "(Không có lịch sử)"

    try:
        role = model_role or settings.question_rewriter_model
        chain = build_rewriter_chain(role)
        from app.core.tracing import get_langfuse_handler, langfuse_context

        trace_ctx = {"trace_id": trace_id} if trace_id else {}
        if parent_span_id and trace_ctx:
            trace_ctx["parent_span_id"] = parent_span_id

        handler = get_langfuse_handler(trace_context=trace_ctx if trace_ctx else None)
        callbacks = [handler] if handler else []
        with langfuse_context(tags=["rag", "question-rewrite"]):
            async with asyncio.timeout(settings.question_rewriter_timeout_seconds):
                rewritten = await chain.ainvoke(
                    {"history": history_text, "question": question},
                    config={"callbacks": callbacks},
                )
        clean = rewritten.strip().strip('"').strip("'")
        max_length = min(2000, max(200, len(question) * 4))
        if not clean or len(clean) > max_length:
            logger.warning("Question rewrite returned empty or oversized output; using original question")
            return RewriteResult(text=question, failed=True)
        if question.rstrip().endswith(("?", "？")) and not clean.rstrip().endswith(("?", "？")):
            if len(clean) >= max_length:
                return RewriteResult(text=question, failed=True)
            clean = clean.rstrip() + "?"
        if unicodedata.normalize("NFC", clean).casefold() in {
            "i don't know", "i cannot answer", "i can't answer", "không biết", "không thể trả lời"
        }:
            return RewriteResult(text=question, failed=True)
    except Exception as exc:
        logger.warning("Question rewrite failed, falling back to original: %s", exc)
        return RewriteResult(text=question, failed=True)

    accent_restored = bool(needs_accent_restoration and not clean.isascii())
    coref_resolved = _normalize_for_compare(clean) != _normalize_for_compare(question)
    return RewriteResult(
        text=clean,
        accent_restored=accent_restored,
        coref_resolved=coref_resolved,
    )


async def rewrite_question_if_needed(
    question: str,
    history: list[dict],
    model_role: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> str:
    """Backward-compatible wrapper returning only the rewritten text."""
    result = await rewrite_question(
        question,
        history,
        model_role=model_role,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
    )
    return result.text
