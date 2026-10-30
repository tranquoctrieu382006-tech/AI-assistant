"""Context-aware question rewriter for multi-turn RAG conversations and diacritic restoration."""
from __future__ import annotations

import logging
import re
import unicodedata
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.core.llm import get_rag_llm
from app.core.session_store import trim_history
from app.core.text_utils import is_probably_unaccented_vietnamese

logger = logging.getLogger(__name__)

# Matches conversational follow-up cues in both accented and unaccented forms
_FOLLOWUP_CUES = re.compile(
    r"\b(?:"
    r"còn|con|thế|the|vậy|vay|đó|do|này|nay|nó|no|họ|ho|ấy|ay|kia|"
    r"chúng|chung|chúng nó|chung no|anh ấy|anh ay|chị ấy|chi ay|bạn ấy|ban ay|"
    r"nữa|nua|trên|tren|như vậy|nhu vay|trường hợp ấy|truong hop ay"
    r")\b",
    re.IGNORECASE,
)

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


async def rewrite_question_if_needed(
    question: str,
    history: list[dict],
    model_role: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> str:
    """
    Rewrites user question given recent dialogue history and restores diacritics if unaccented.
    If question is already accented and history is empty (or standalone), skips LLM call immediately.
    """
    needs_accent_restoration = is_probably_unaccented_vietnamese(question)
    trimmed = trim_history(history, max_turns=5) if history else []

    needs_context_rewrite = bool(
        trimmed and (len(question.split()) < 4 or _FOLLOWUP_CUES.search(question))
    )

    if not needs_accent_restoration and not needs_context_rewrite:
        return question

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
            return question
        if question.rstrip().endswith(("?", "？")) and not clean.rstrip().endswith(("?", "？")):
            if len(clean) >= max_length:
                return question
            clean = clean.rstrip() + "?"
        if unicodedata.normalize("NFC", clean).casefold() in {
            "i don't know", "i cannot answer", "i can't answer", "không biết", "không thể trả lời"
        }:
            return question
        return clean
    except Exception as exc:
        logger.warning("Question rewrite failed, falling back to original: %s", exc)
        return question
