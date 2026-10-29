"""In-memory session history storage and token trimmer for multi-turn chat."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# In-memory dictionary: session_id -> list of message dicts
_SESSIONS: OrderedDict[str, list[dict]] = OrderedDict()
_SESSION_LAST_ACTIVITY: dict[str, float] = {}
_MAX_SESSIONS = 1000
_MAX_MESSAGES_PER_SESSION = 40
_SESSION_TTL_SECONDS = 30 * 60

# Corpus deletes hit PostgreSQL, so they must never run on the event loop.
# Async callers submit them to a small dedicated executor; sync callers (tests,
# CLI scripts) keep the previous synchronous behaviour.
_PENDING_CLEANUPS: set[concurrent.futures.Future] = set()
_PENDING_CLEANUPS_LOCK = threading.Lock()
_CLEANUP_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="session-corpus-cleanup")


def _cleanup_session_corpus(session_id: str) -> None:
    """Best-effort removal of uploaded chunks when a session is evicted.

    Session histories live in process memory while their uploaded documents live
    in PostgreSQL. Without this hook the session corpus outlives the session and
    keeps growing across TTL expiry and LRU eviction.
    """
    try:
        from app.core import vector_store

        deleted = vector_store.delete_corpus(f"session_{session_id}")
    except Exception as exc:
        logger.warning("Không thể dọn corpus cho session '%s': %s", session_id, exc)
        return
    if not deleted:
        return
    try:
        from app.core.retriever import clear_bm25_cache

        clear_bm25_cache(corpus=f"session_{session_id}")
    except Exception as exc:
        logger.warning("Không thể xoá cache BM25 sau khi dọn session '%s': %s", session_id, exc)


def _dispatch_corpus_cleanup(session_id: str) -> None:
    """Run the PostgreSQL delete off the event loop when one is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _cleanup_session_corpus(session_id)
        return

    future = _CLEANUP_EXECUTOR.submit(_cleanup_session_corpus, session_id)
    with _PENDING_CLEANUPS_LOCK:
        _PENDING_CLEANUPS.add(future)

    def _done(completed: concurrent.futures.Future) -> None:
        with _PENDING_CLEANUPS_LOCK:
            _PENDING_CLEANUPS.discard(completed)
        try:
            completed.result()
        except Exception as exc:  # pragma: no cover - cleanup already logs
            logger.warning("Tác vụ dọn corpus session thất bại: %s", exc)

    future.add_done_callback(_done)


def wait_for_pending_cleanups(timeout: float = 5.0) -> None:
    """Wait for background corpus deletes (tests and graceful shutdown)."""
    with _PENDING_CLEANUPS_LOCK:
        pending = list(_PENDING_CLEANUPS)
    if pending:
        concurrent.futures.wait(pending, timeout=timeout)


def _expire_sessions(now: float) -> None:
    # Session order is maintained from least to most recently active, so only
    # inspect entries that can actually expire instead of scanning all sessions.
    while _SESSIONS:
        session_id = next(iter(_SESSIONS))
        last_active = _SESSION_LAST_ACTIVITY.get(session_id, 0.0)
        if now - last_active < _SESSION_TTL_SECONDS:
            break
        _SESSIONS.pop(session_id, None)
        _SESSION_LAST_ACTIVITY.pop(session_id, None)
        _dispatch_corpus_cleanup(session_id)


def _evict_if_needed(session_id: str) -> None:
    if len(_SESSIONS) < _MAX_SESSIONS or session_id in _SESSIONS:
        return
    # OrderedDict is maintained as least- to most-recently-used.
    oldest_key = next(iter(_SESSIONS))
    _SESSIONS.pop(oldest_key, None)
    _SESSION_LAST_ACTIVITY.pop(oldest_key, None)
    _dispatch_corpus_cleanup(oldest_key)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = Field(default_factory=time.time)


def touch_session(session_id: str | None) -> None:
    """Register activity for a session even before its first question.

    Uploads create a session corpus in PostgreSQL. Registering the session at
    upload time means the normal TTL/LRU cleanup hooks can eventually remove
    that corpus instead of leaving it orphaned.
    """
    if not session_id:
        return
    now = time.monotonic()
    _expire_sessions(now)
    _evict_if_needed(session_id)
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    _SESSIONS.move_to_end(session_id)
    _SESSION_LAST_ACTIVITY[session_id] = now


def get_history(session_id: str | None) -> list[dict]:
    """Retrieve message history for a given session."""
    if not session_id:
        return []
    now = time.monotonic()
    _expire_sessions(now)
    if session_id not in _SESSIONS:
        return []
    _SESSIONS.move_to_end(session_id)
    _SESSION_LAST_ACTIVITY[session_id] = now
    return list(_SESSIONS[session_id])


def add_turn(session_id: str | None, question: str, answer: str) -> None:
    """Append a user question and assistant answer to the session history."""
    if not session_id:
        return

    now = time.monotonic()
    _expire_sessions(now)
    _evict_if_needed(session_id)

    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []

    _SESSIONS[session_id].append({"role": "user", "content": question, "timestamp": time.time()})
    _SESSIONS[session_id].append({"role": "assistant", "content": answer, "timestamp": time.time()})
    if len(_SESSIONS[session_id]) > _MAX_MESSAGES_PER_SESSION:
        del _SESSIONS[session_id][:-_MAX_MESSAGES_PER_SESSION]
    _SESSIONS.move_to_end(session_id)
    _SESSION_LAST_ACTIVITY[session_id] = now


def clear_session(session_id: str) -> None:
    """Clear history for a specific session and schedule corpus cleanup."""
    existed = session_id in _SESSIONS
    _SESSIONS.pop(session_id, None)
    _SESSION_LAST_ACTIVITY.pop(session_id, None)
    if existed:
        _dispatch_corpus_cleanup(session_id)


def clear_all_sessions() -> None:
    """Clear all sessions (useful for unit tests)."""
    _SESSIONS.clear()
    _SESSION_LAST_ACTIVITY.clear()


def sweep_orphan_session_corpora() -> int:
    """Delete session corpora that no in-memory session owns.

    Session ownership lives only in process memory, so after a restart every
    ``session_*`` corpus is orphaned by definition. Running this best-effort
    sweep at startup prevents those rows from accumulating forever.
    """
    try:
        from app.core import vector_store

        corpora = vector_store.list_corpora(prefix="session_")
    except Exception as exc:
        logger.warning("Không thể quét corpus session mồ côi: %s", exc)
        return 0

    active = {f"session_{session_id}" for session_id in list(_SESSIONS)}
    removed = 0
    for corpus in corpora:
        if corpus in active:
            continue
        try:
            if vector_store.delete_corpus(corpus):
                removed += 1
        except Exception as exc:
            logger.warning("Không thể xoá corpus mồ côi '%s': %s", corpus, exc)
    if removed:
        try:
            from app.core.retriever import clear_bm25_cache

            clear_bm25_cache()
        except Exception as exc:
            logger.warning("Không thể xoá cache BM25 sau khi quét corpus: %s", exc)
    return removed


def trim_history(
    history: list[dict],
    max_turns: int = 5,
    max_chars: int = 8000,
) -> list[dict]:
    """
    Trim history to keep at most `max_turns` (each turn = 1 user + 1 assistant),
    bounded by total character budget.
    """
    if not history:
        return []

    if max_turns <= 0 or max_chars <= 0:
        return []

    # Keep at most max_turns * 2 messages, then consume the budget from newest
    # to oldest so an oversized old answer cannot evict the latest user prompt.
    trimmed = history[-(max_turns * 2):]
    kept: list[dict] = []
    used = 0
    for message in reversed(trimmed):
        content = str(message.get("content", ""))
        remaining = max_chars - used
        if len(content) <= remaining:
            kept.append(message)
            used += len(content)
            continue
        if not kept and remaining:
            marker = "[...nội dung trước đã được lược bớt...]\n"
            if remaining > len(marker):
                content = marker + content[-(remaining - len(marker)):]
            else:
                content = content[-remaining:]
            kept.append({**message, "content": content})
        break
    return list(reversed(kept))
