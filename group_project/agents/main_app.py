from __future__ import annotations

import sys
from pathlib import Path

import chainlit as cl

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.memory_store import append_turn, get_history, get_value, reset_history, set_value
from agents.orchestrator import build_orchestrator
from agents.source_view import format_citations, format_sources_panel


APP_TITLE = "RAG Chatbot"
APP_SUBTITLE = "Khung chat nhóm - sẵn sàng để cắm RAG pipeline."


@cl.on_chat_start
async def on_chat_start() -> None:
    if get_value("orchestrator") is None:
        set_value("orchestrator", build_orchestrator())
    reset_history()
    await cl.Message(
        content=f"**{APP_TITLE}**\n\n{APP_SUBTITLE}",
        actions=[cl.Action(name="reset_chat", payload={}, label="Reset chat")],
    ).send()


@cl.action_callback("reset_chat")
async def reset_chat_action(action: cl.Action) -> None:
    await action.remove()
    reset_history()
    await cl.Message(content="Da reset lich su chat.").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    orchestrator = get_value("orchestrator") or build_orchestrator()
    history = get_history()
    append_turn("user", message.content)
    response = orchestrator.answer(message.content, history)
    response_dict = response.to_dict()
    append_turn("assistant", response_dict["answer"])

    await cl.Message(content=response_dict["answer"]).send()
    if response_dict["citations"]:
        await cl.Message(content=f"**Citation**\n{format_citations(response.citations)}").send()
    if response_dict["sources"]:
        await cl.Message(content=f"**Sources**\n{format_sources_panel(response.sources)}").send()
