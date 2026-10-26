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
    # Session order is maintained from least to most recently active, so only
    # inspect entries that can actually expire instead of scanning all sessions.
    while _SESSIONS:
        session_id = next(iter(_SESSIONS))
        last_active = _SESSION_LAST_ACTIVITY.get(session_id, 0.0)
        if now - last_active < _SESSION_TTL_SECONDS:
            break
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
