from __future__ import annotations

import sys
from pathlib import Path

import chainlit as cl

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.memory_store import append_turn, get_history, get_value, reset_history, set_value
from agents.orchestrator import build_orchestrator
from agents.session_bridge import build_backend_history, format_session_overview
from agents.source_view import format_citations, format_sources_sidebar


APP_TITLE = "RAG Chatbot"
APP_SUBTITLE = "Khung chat nhóm về phòng chống ma túy - sẵn sàng để cắm RAG pipeline."
BANNER_PATH = "/public/rag-hero.gif"

QUICK_PROMPTS = [
    "Những dấu hiệu nào thường được nhắc đến trong tài liệu về phòng chống ma túy?",
    "Nguồn nào trong tập tài liệu nói rõ biện pháp phòng chống nhất?",
    "Nếu không đủ evidence, chatbot nên trả lời thế nào để an toàn?",
]


def build_actions() -> list[cl.Action]:
    return [
        cl.Action(
            name="quick_question",
            payload={"value": QUICK_PROMPTS[0]},
            label="Câu mẫu 1",
            tooltip="Mở câu hỏi demo đầu tiên",
            icon="sparkles",
        ),
        cl.Action(
            name="quick_question",
            payload={"value": QUICK_PROMPTS[1]},
            label="Câu mẫu 2",
            tooltip="Mở câu hỏi demo thứ hai",
            icon="book-open",
        ),
        cl.Action(
            name="quick_question",
            payload={"value": QUICK_PROMPTS[2]},
            label="Câu mẫu 3",
            tooltip="Mở câu hỏi demo thứ ba",
            icon="shield-alert",
        ),
        cl.Action(
            name="reset_chat",
            payload={},
            label="Reset chat",
            tooltip="Xóa lịch sử hiện tại",
            icon="rotate-ccw",
        ),
    ]


def build_intro_message() -> str:
    return (
        f"![{APP_TITLE}]({BANNER_PATH})\n\n"
        f"{APP_SUBTITLE}\n\n"
        "Chatbot này tập trung vào câu trả lời có căn cứ, câu hỏi follow-up, và hiển thị nguồn trích dẫn rõ ràng.\n\n"
        "## Demo Scope\n"
        "- Answer có citation rõ ràng.\n"
        "- Follow-up bằng memory.\n"
        "- Source documents hiển thị riêng.\n\n"
        "## Cách dùng nhanh\n"
        "1. Bấm một câu mẫu để xem luồng demo.\n"
        "2. Hoặc nhập câu hỏi của bạn trực tiếp.\n"
        "3. Mỗi câu trả lời sẽ được tách thành answer, citation và nguồn."
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    if get_value("orchestrator") is None:
        set_value("orchestrator", build_orchestrator())
    reset_history()
    await cl.Message(
        content=build_intro_message(),
        actions=build_actions(),
        metadata={"surface": "welcome", "app": APP_TITLE},
    ).send()
    await cl.Message(
        content=format_session_overview(get_history()),
        author="RAG Assistant",
        tags=["session"],
    ).send()


@cl.action_callback("reset_chat")
async def reset_chat_action(action: cl.Action) -> None:
    await action.remove()
    reset_history()
    await cl.Message(content="Đã reset lịch sử chat.", actions=build_actions()).send()


@cl.action_callback("quick_question")
async def quick_question_action(action: cl.Action) -> None:
    question = action.payload.get("value", "").strip()
    await action.remove()
    if not question:
        return
    await handle_question(question, is_quick_prompt=True)


async def render_response(response) -> None:
    response_dict = response.to_dict()
    answer = response_dict.get("answer", "").strip()
    metadata = response_dict.get("metadata", {})
    retrieval_mode = metadata.get("retrieval_mode")
    used_memory = metadata.get("used_memory")

    if not answer:
        answer = "Khong co noi dung tra loi."

    sidebar_text = format_sources_sidebar(
        response.sources,
        citations=response.citations,
        retrieval_mode=retrieval_mode,
        used_memory=used_memory,
    )

    await cl.Message(
        content=answer,
        author="RAG Assistant",
        metadata=metadata,
        tags=["answer", "rag"],
        elements=[
            cl.Text(
                name="Source Panel",
                content=sidebar_text,
                display="side",
                language="markdown",
            )
        ],
    ).send()

    if response.citations:
        await cl.Message(
            content=format_citations(response.citations),
            author="RAG Assistant",
            tags=["citation"],
        ).send()

    if response.sources:
        await cl.Message(
            content=f"**Sources** ready in the side panel ({len(response.sources)} docs).",
            author="RAG Assistant",
            tags=["sources"],
        ).send()


async def handle_question(question: str, is_quick_prompt: bool = False) -> None:
    orchestrator = get_value("orchestrator") or build_orchestrator()
    history = build_backend_history(get_history())
    append_turn("user", question)

    if is_quick_prompt:
        await cl.Message(
            content=f"**Câu hỏi mẫu**\n\n{question}",
            author="You",
            tags=["quick-prompt"],
        ).send()

    response = orchestrator.answer(question, history)
    append_turn("assistant", response.answer)
    await render_response(response)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await handle_question(message.content)
