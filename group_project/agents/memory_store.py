from __future__ import annotations

from typing import Any

try:
    import chainlit as cl
except Exception:  # pragma: no cover
    cl = None  # type: ignore[assignment]


HISTORY_KEY = "chat_history"


def get_history() -> list[dict[str, str]]:
    if cl is None:
        return []
    return cl.user_session.get(HISTORY_KEY, []) or []


def append_turn(role: str, content: str) -> None:
    if cl is None:
        return
    history = get_history()
    history.append({"role": role, "content": content})
    cl.user_session.set(HISTORY_KEY, history)


def reset_history() -> None:
    if cl is None:
        return
    cl.user_session.set(HISTORY_KEY, [])


def set_value(key: str, value: Any) -> None:
    if cl is None:
        return
    cl.user_session.set(key, value)


def get_value(key: str, default: Any = None) -> Any:
    if cl is None:
        return default
    return cl.user_session.get(key, default)

