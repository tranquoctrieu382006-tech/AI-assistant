"""Context-aware question rewriter for multi-turn RAG conversations."""
from __future__ import annotations

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.core.llm import get_llm
from app.core.session_store import trim_history

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """Bạn là chuyên viên làm rõ câu hỏi trong hệ thống hỏi đáp tài liệu nội bộ.
Dựa vào LỊCH SỬ HỘI THOẠI và CÂU HỎI MỚI, hãy diễn giải câu hỏi mới thành một câu hỏi ĐỘC LẬP, RÕ RÀNG bằng tiếng Việt.

Quy tắc bắt buộc:
1. Nếu câu hỏi mới đã rõ ràng, đầy đủ ngữ cảnh hoặc bắt đầu một chủ đề mới: GIỮ NGUYÊN câu hỏi.
2. Nếu câu hỏi mới dùng đại từ thay thế hoặc bị tỉnh lược (ví dụ: "thế còn hàng lỗi?", "phải gửi trước bao lâu?", "ai ký?"): bổ sung ngữ cảnh từ lượt trước để câu hỏi có thể hiểu được khi đứng một mình.
3. CHỈ xuất ra duy nhất câu hỏi đã làm rõ. KHÔNG giải thích, KHÔNG chào hỏi, KHÔNG trả lời câu hỏi.
"""


def build_rewriter_chain(model_role: str = "local"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM_PROMPT),
        ("human", "LỊCH SỬ HỘI THOẠI:\n{history}\n\nCÂU HỎI MỚI: {question}\n\nCÂU HỎI ĐỘC LẬP:"),
    ])
    llm = get_llm(model_role, temperature=0.0)
    return prompt | llm | StrOutputParser()


async def rewrite_question_if_needed(
    question: str,
    history: list[dict],
    model_role: str | None = None,
) -> str:
    """
    Rewrites user question given recent dialogue history.
    If history is empty, skips LLM call and returns original question immediately.
    """
    if not history:
        return question

    trimmed = trim_history(history, max_turns=5)
    if not trimmed:
        return question

    history_lines = []
    for msg in trimmed:
        role_label = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        history_lines.append(f"{role_label}: {msg['content']}")
    history_text = "\n".join(history_lines)

    role = model_role or settings.rag_model
    chain = build_rewriter_chain(role)
    try:
        rewritten = await chain.ainvoke({"history": history_text, "question": question})
        clean = rewritten.strip().strip('"').strip("'")
        return clean or question
    except Exception as exc:
        logger.warning("Question rewrite failed, falling back to original: %s", exc)
        return question
