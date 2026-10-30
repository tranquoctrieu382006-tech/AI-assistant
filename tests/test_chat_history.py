"""Unit tests for multi-turn chat history and question rewriter (TASK-AI11)."""
import asyncio
import threading
import time
import pytest
from unittest.mock import AsyncMock, patch

from app.core import session_store
from app.core import question_rewriter
from app.core.question_rewriter import (
    RewriteResult,
    rewrite_question,
    rewrite_question_if_needed,
    trim_history,
)


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

    with patch("app.core.vector_store.delete_corpus", return_value=0):
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
    with patch("app.core.vector_store.delete_corpus", return_value=0) as delete_corpus:
        assert session_store.get_history("old") == []
    delete_corpus.assert_called_once_with("session_old")
    assert "old" not in session_store._SESSIONS
    assert list(session_store._SESSIONS) == ["recent"]


def test_session_eviction_uses_least_recently_used(monkeypatch):
    session_store.clear_all_sessions()
    monkeypatch.setattr(session_store, "_MAX_SESSIONS", 2)
    session_store.add_turn("a", "q", "a")
    session_store.add_turn("b", "q", "a")
    session_store.get_history("a")
    with patch("app.core.vector_store.delete_corpus", return_value=0) as delete_corpus:
        session_store.add_turn("c", "q", "a")
    delete_corpus.assert_called_once_with("session_b")
    assert list(session_store._SESSIONS) == ["a", "c"]


def test_session_eviction_clears_bm25_cache_when_corpus_had_rows(monkeypatch):
    """Khi corpus session bị xoá thật, index BM25 đã cache cũng phải bị xoá."""
    session_store.clear_all_sessions()
    monkeypatch.setattr(session_store, "_MAX_SESSIONS", 1)
    session_store.add_turn("a", "q", "a")
    with (
        patch("app.core.vector_store.delete_corpus", return_value=3),
        patch("app.core.retriever.clear_bm25_cache") as clear_cache,
    ):
        session_store.add_turn("b", "q", "a")
    clear_cache.assert_called_once_with(corpus="session_a")


def test_touch_session_registers_upload_only_session():
    """Upload-only sessions must be known to the TTL/LRU cleanup hooks."""
    session_store.clear_all_sessions()

    session_store.touch_session("upload-only")

    assert "upload-only" in session_store._SESSIONS
    assert session_store._SESSION_LAST_ACTIVITY["upload-only"] > 0
    assert session_store.get_history("upload-only") == []


def test_sweep_orphan_session_corpora_keeps_unverifiable_corpora(monkeypatch):
    """Không xoá dữ liệu khi không kiểm tra được last_seen."""
    session_store.clear_all_sessions()
    deleted: list[str] = []

    with (
        patch(
            "app.core.vector_store.list_corpora",
            return_value=["session_left_over"],
        ),
        patch(
            "app.core.vector_store.filter_stale_corpora",
            side_effect=RuntimeError("activity table unavailable"),
        ),
        patch(
            "app.core.vector_store.delete_corpus",
            side_effect=lambda corpus: deleted.append(corpus) or 1,
        ),
    ):
        removed = session_store.sweep_orphan_session_corpora()

    assert removed == 0
    assert deleted == []


def test_sweep_orphan_session_corpora_keeps_active_sessions(monkeypatch):
    session_store.clear_all_sessions()
    session_store.add_turn("active", "q", "a")
    deleted: list[str] = []

    with (
        patch(
            "app.core.vector_store.list_corpora",
            return_value=["session_active", "session_left_over"],
        ),
        patch(
            "app.core.vector_store.filter_stale_corpora",
            return_value=["session_left_over"],
        ),
        patch(
            "app.core.vector_store.delete_corpus",
            side_effect=lambda corpus: deleted.append(corpus) or 1,
        ),
        patch("app.core.retriever.clear_bm25_cache") as clear_cache,
    ):
        removed = session_store.sweep_orphan_session_corpora()

    assert removed == 1
    assert deleted == ["session_left_over"]
    clear_cache.assert_called_once_with()


@pytest.mark.asyncio
async def test_session_expiry_dispatches_delete_off_event_loop(monkeypatch):
    """A slow PostgreSQL DELETE must not run inside the request event loop."""
    session_store.clear_all_sessions()
    monkeypatch.setattr(session_store, "_SESSION_TTL_SECONDS", 0)
    session_store.add_turn("expired", "q", "a")

    started = threading.Event()
    release = threading.Event()

    def blocking_delete(corpus: str) -> int:
        started.set()
        release.wait(5)
        return 0

    try:
        with patch("app.core.vector_store.delete_corpus", side_effect=blocking_delete):
            before = time.monotonic()
            assert session_store.get_history("expired") == []
            elapsed = time.monotonic() - before
            assert elapsed < 1.0
            assert await asyncio.to_thread(started.wait, 2)
    finally:
        release.set()
        session_store.wait_for_pending_cleanups(5)


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
async def test_long_followup_with_pronoun_is_rewritten():
    history = [
        {"role": "user", "content": "Chính sách thử việc thế nào?"},
        {"role": "assistant", "content": "Áp dụng theo hợp đồng thử việc."},
    ]
    question = "Nó áp dụng cho nhân viên thử việc không?"
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Chính sách đó áp dụng cho nhân viên thử việc không?"
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain) as build:
        rewritten = await rewrite_question_if_needed(question, history)
    build.assert_called_once()
    assert rewritten.startswith("Chính sách đó")


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
async def test_rewrite_skips_medium_self_contained_question_without_followup_cue():
    history = [
        {"role": "user", "content": "Tồn kho của sản phẩm A còn bao nhiêu?"},
        {"role": "assistant", "content": "Còn 12 sản phẩm."},
    ]
    question = "Doanh thu năm 2025 là bao nhiêu?"
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


def test_rewrite_timeout_falls_back_to_original_question(monkeypatch):
    history = [
        {"role": "user", "content": "Chính sách đổi trả thế nào?"},
        {"role": "assistant", "content": "Trong 7 ngày."},
    ]
    async def slow_rewrite(*args, **kwargs):
        import asyncio
        await asyncio.sleep(0.1)

    monkeypatch.setattr(question_rewriter.settings, "question_rewriter_timeout_seconds", 0.01)
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = slow_rewrite
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain):
        rewritten = asyncio.run(rewrite_question_if_needed("Thế còn hàng lỗi?", history))
    assert rewritten == "Thế còn hàng lỗi?"


@pytest.mark.asyncio
async def test_rewrite_triggers_on_unaccented_vietnamese_even_without_history():
    """Câu hỏi không dấu ở lượt đầu tiên (history=[]) vẫn được kích hoạt rewriter để khôi phục dấu."""
    q_unaccented = "quy trinh tra hang nhu the nao"
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Quy trình trả hàng như thế nào?"
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain) as build_mock:
        rewritten = await rewrite_question_if_needed(q_unaccented, history=[])
        build_mock.assert_called_once()
        assert rewritten == "Quy trình trả hàng như thế nào?"


@pytest.mark.asyncio
async def test_unaccented_followup_cue_triggers_rewrite():
    """Đại từ thay thế dạng không dấu (the con, vay thi, do la...) phải khớp _FOLLOWUP_CUES."""
    history = [
        {"role": "user", "content": "Chính sách đổi trả thế nào?"},
        {"role": "assistant", "content": "Trong 7 ngày."},
    ]
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Chính sách đổi trả đối với hàng lỗi như thế nào?"
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain) as build_mock:
        rewritten = await rewrite_question_if_needed("the con hang loi thi sao", history)
        build_mock.assert_called_once()
        assert "hàng lỗi" in rewritten


@pytest.mark.parametrize(
    "question",
    [
        "Hiện nay công ty có bao nhiêu nhân viên?",
        "Dùng chung tài khoản được không?",
        "Do lỗi giao hàng nên tôi muốn đổi trả",
    ],
)
@pytest.mark.asyncio
async def test_ambiguous_cue_does_not_trigger_rewrite(question):
    """Các phản ví dụ có dấu không được coi là câu nối tiếp chỉ vì chứa cue."""
    history = [
        {"role": "user", "content": "Chính sách đổi trả thế nào?"},
        {"role": "assistant", "content": "Trong 7 ngày."},
    ]
    with patch("app.core.question_rewriter.build_rewriter_chain") as build_chain:
        result = await rewrite_question(question, history)
    build_chain.assert_not_called()
    assert result == RewriteResult(text=question)


@pytest.mark.asyncio
async def test_strong_cue_still_triggers_rewrite_for_accented_followup():
    history = [
        {"role": "user", "content": "Chính sách đổi trả thế nào?"},
        {"role": "assistant", "content": "Trong 7 ngày."},
    ]
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Chính sách đổi trả đối với hàng lỗi như thế nào?"
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain):
        result = await rewrite_question("Thế còn hàng lỗi?", history)
    assert result.text.startswith("Chính sách đổi trả")
    assert result.accent_restored is False
    assert result.coref_resolved is True


@pytest.mark.asyncio
async def test_rewrite_flags_accent_only_change_is_not_coref():
    """Khôi phục dấu thuần túy không được coi là giải quyết đại từ."""
    raw = "quy trinh tra hang nhu the nao"
    mock_chain = AsyncMock()
    # Khác hoa/thường và dấu câu, nhưng nội dung từ không đổi.
    mock_chain.ainvoke.return_value = "Quy trình trả hàng như thế nào?"
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain):
        result = await rewrite_question(raw, history=[])
    assert result.accent_restored is True
    assert result.coref_resolved is False
    assert result.failed is False


@pytest.mark.asyncio
async def test_rewrite_flags_coref_when_content_changes():
    raw = "quy trinh tra hang nhu the nao"
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Quy trình trả hàng cho đơn hàng mua online như thế nào?"
    with patch("app.core.question_rewriter.build_rewriter_chain", return_value=mock_chain):
        result = await rewrite_question(raw, history=[])
    assert result.accent_restored is True
    assert result.coref_resolved is True


def test_text_utils_and_unaccented_detection():
    from app.core.text_utils import is_probably_unaccented_vietnamese, strip_accents

    assert is_probably_unaccented_vietnamese("quy trinh tra hang nhu the nao") is True
    assert is_probably_unaccented_vietnamese("thoi han doi tra la bao lau") is True
    assert is_probably_unaccented_vietnamese("What is the refund policy?") is False
    assert is_probably_unaccented_vietnamese("ORD-12345") is False
    assert is_probably_unaccented_vietnamese("Chính sách bảo hành?") is False  # Already accented

    # Kiểm tra xử lý chữ đ/Đ
    assert strip_accents("Đơn hàng đổi trả") == "don hang doi tra"
    assert strip_accents("ĐỒNG NAI") == "dong nai"
