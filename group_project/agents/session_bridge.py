from __future__ import annotations

from textwrap import shorten
from typing import Any


def sanitize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    clean_history: list[dict[str, str]] = []
    for turn in history or []:
        role = (turn.get("role") or "").strip()
        content = (turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        clean_history.append({"role": role, "content": content})
    return clean_history


def build_backend_history(history: list[dict[str, str]] | None, max_turns: int = 8) -> list[dict[str, str]]:
    clean_history = sanitize_history(history)
    if max_turns <= 0:
        return clean_history
    return clean_history[-max_turns:]


def build_session_overview(history: list[dict[str, str]] | None) -> dict[str, Any]:
    clean_history = sanitize_history(history)
    user_turns = [turn["content"] for turn in clean_history if turn["role"] == "user"]
    assistant_turns = [turn["content"] for turn in clean_history if turn["role"] == "assistant"]

    return {
        "total_turns": len(clean_history),
        "user_turns": len(user_turns),
        "assistant_turns": len(assistant_turns),
        "last_user": user_turns[-1] if user_turns else "",
        "last_assistant": assistant_turns[-1] if assistant_turns else "",
        "memory_ready": len(user_turns) >= 1,
        "follow_up_ready": len(user_turns) >= 2,
    }


def format_session_overview(history: list[dict[str, str]] | None) -> str:
    overview = build_session_overview(history)
    last_user = overview["last_user"]
    last_assistant = overview["last_assistant"]

    lines = [
        "### Session Ready",
        f"- Turns: {overview['user_turns']} user / {overview['assistant_turns']} assistant",
        f"- Memory: {'enabled' if overview['memory_ready'] else 'idle'}",
        f"- Follow-up: {'ready' if overview['follow_up_ready'] else 'waiting for the first two turns'}",
    ]
    if last_user:
        lines.append(f"- Last user: {shorten(last_user, width=72, placeholder='...')}")
    if last_assistant:
        lines.append(f"- Last answer: {shorten(last_assistant, width=72, placeholder='...')}")
    return "\n".join(lines)
