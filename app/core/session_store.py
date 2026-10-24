"""In-memory session history storage and token trimmer for multi-turn chat."""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Literal
from pydantic import BaseModel, Field

# In-memory dictionary: session_id -> list of message dicts
_SESSIONS: OrderedDict[str, list[dict]] = OrderedDict()
_SESSION_LAST_ACTIVITY: dict[str, float] = {}
_MAX_SESSIONS = 1000
_MAX_MESSAGES_PER_SESSION = 40
_SESSION_TTL_SECONDS = 30 * 60


def _expire_sessions(now: float) -> None:
    expired = [
        session_id for session_id, last_active in _SESSION_LAST_ACTIVITY.items()
        if now - last_active >= _SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        _SESSIONS.pop(session_id, None)
        _SESSION_LAST_ACTIVITY.pop(session_id, None)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = Field(default_factory=time.time)


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
    if len(_SESSIONS) >= _MAX_SESSIONS and session_id not in _SESSIONS:
        # OrderedDict is maintained as least- to most-recently-used.
        oldest_key = next(iter(_SESSIONS))
        _SESSIONS.pop(oldest_key, None)
        _SESSION_LAST_ACTIVITY.pop(oldest_key, None)

    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []

    _SESSIONS[session_id].append({"role": "user", "content": question, "timestamp": time.time()})
    _SESSIONS[session_id].append({"role": "assistant", "content": answer, "timestamp": time.time()})
    if len(_SESSIONS[session_id]) > _MAX_MESSAGES_PER_SESSION:
        del _SESSIONS[session_id][:-_MAX_MESSAGES_PER_SESSION]
    _SESSIONS.move_to_end(session_id)
    _SESSION_LAST_ACTIVITY[session_id] = now


def clear_session(session_id: str) -> None:
    """Clear history for a specific session."""
    _SESSIONS.pop(session_id, None)
    _SESSION_LAST_ACTIVITY.pop(session_id, None)


def clear_all_sessions() -> None:
    """Clear all sessions (useful for unit tests)."""
    _SESSIONS.clear()
    _SESSION_LAST_ACTIVITY.clear()


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

    # Keep at most max_turns * 2 messages
    trimmed = history[-(max_turns * 2):]

    # Enforce character budget from newest to oldest
    total_len = 0
    kept_from_idx = 0
    for idx in range(len(trimmed) - 1, -1, -1):
        msg_len = len(trimmed[idx].get("content", ""))
        if total_len + msg_len > max_chars:
            kept_from_idx = idx + 1
            break
        total_len += msg_len

    return trimmed[kept_from_idx:]
