"""Unit tests for multi-turn chat history and question rewriter (TASK-AI11)."""
import pytest
from unittest.mock import AsyncMock, patch

from app.core import session_store
from app.core.question_rewriter import rewrite_question_if_needed, trim_history


def test_session_store_crud():
    session_store.clear_all_sessions()
    session_id = "test-session-1"

    assert session_store.get_history(session_id) == []

    session_store.add_turn(session_id, "Câu hỏi 1", "Trả lời 1")
    history = session_store.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Câu hỏi 1"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Trả lời 1"

    session_store.add_turn(session_id, "Câu hỏi 2", "Trả lời 2")
    history = session_store.get_history(session_id)
    assert len(history) == 4

    session_store.clear_session(session_id)
    assert session_store.get_history(session_id) == []


def test_trim_history_preserves_most_recent_turns():
    history = [
        {"role": "user", "content": f"Q{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"A{i}"}
        for i in range(20)
    ]
    # max_turns=3 -> retains last 6 messages
    trimmed = trim_history(history, max_turns=3)
    assert len(trimmed) == 6
    assert trimmed[-1]["content"] == "A19"
    assert trimmed[0]["content"] == "Q14"


@pytest.mark.asyncio
async def test_rewrite_skipped_when_history_empty():
    q = "Chính sách đổi trả hàng như thế nào?"
    res = await rewrite_question_if_needed(q, history=[])
    assert res == q


@pytest.mark.asyncio
async def test_rewrite_calls_llm_when_history_present():
    history = [
        {"role": "user", "content": "Chính sách đổi trả hàng như thế nào?"},
        {"role": "assistant", "content": "Trong vòng 7 ngày lịch kể từ ngày nhận hàng."},
    ]
    follow_up = "Thế còn hàng lỗi?"

    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Chính sách đổi trả hàng đối với sản phẩm bị lỗi như thế nào?"

    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain):
        rewritten = await rewrite_question_if_needed(follow_up, history, model_role="local")

    assert rewritten == "Chính sách đổi trả hàng đối với sản phẩm bị lỗi như thế nào?"
    mock_chain.ainvoke.assert_awaited_once()
