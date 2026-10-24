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


def test_session_store_keeps_last_40_messages_on_turn_21():
    session_store.clear_all_sessions()
    session_id = "bounded-history"
    for turn in range(21):
        session_store.add_turn(session_id, f"Q{turn}", f"A{turn}")
    history = session_store.get_history(session_id)
    assert len(history) == 40
    assert history[0]["content"] == "Q1"
    assert history[-1]["content"] == "A20"


def test_session_store_expires_inactive_sessions_and_refreshes_lru(monkeypatch):
    session_store.clear_all_sessions()
    monkeypatch.setattr(session_store, "_SESSION_TTL_SECONDS", 60)
    session_store.add_turn("old", "q", "a")
    session_store.add_turn("recent", "q", "a")
    session_store._SESSION_LAST_ACTIVITY["old"] = 0
    assert session_store.get_history("old") == []
    assert "old" not in session_store._SESSIONS
    assert list(session_store._SESSIONS) == ["recent"]


def test_session_eviction_uses_least_recently_used(monkeypatch):
    session_store.clear_all_sessions()
    monkeypatch.setattr(session_store, "_MAX_SESSIONS", 2)
    session_store.add_turn("a", "q", "a")
    session_store.add_turn("b", "q", "a")
    session_store.get_history("a")
    session_store.add_turn("c", "q", "a")
    assert list(session_store._SESSIONS) == ["a", "c"]


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


@pytest.mark.asyncio
async def test_rewriter_uses_its_explicit_model_setting(monkeypatch):
    history = [
        {"role": "user", "content": "Câu hỏi trước?"},
        {"role": "assistant", "content": "Câu trả lời."},
    ]
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Câu hỏi độc lập?"
    monkeypatch.setattr("app.core.question_rewriter.settings.question_rewriter_model", "fast")
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain) as build:
        await rewrite_question_if_needed("Còn ý này?", history)
    build.assert_called_once_with("fast")


@pytest.mark.asyncio
async def test_rewrite_skips_long_self_contained_question_even_with_history():
    history = [
        {"role": "user", "content": "Question one"},
        {"role": "assistant", "content": "Answer one"},
    ]
    question = "Nhân viên cần gửi đơn xin nghỉ phép cho phòng nhân sự trước bao nhiêu ngày?"
    with patch("app.core.question_rewriter.build_rewriter_chain") as build_chain:
        rewritten = await rewrite_question_if_needed(question, history)
    assert rewritten == question
    build_chain.assert_not_called()


@pytest.mark.asyncio
async def test_rewrite_adds_question_mark_instead_of_discarding_valid_rewrite():
    history = [
        {"role": "user", "content": "Chính sách đổi trả thế nào?"},
        {"role": "assistant", "content": "Trong 7 ngày."},
    ]
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Chính sách đổi trả đối với hàng lỗi như thế nào"
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain):
        rewritten = await rewrite_question_if_needed("Thế còn hàng lỗi?", history)
    assert rewritten.endswith("?")
    assert "đối với hàng lỗi" in rewritten
