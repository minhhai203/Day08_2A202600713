from __future__ import annotations

import sys
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.memory_store import append_turn, get_history, get_value, reset_history, set_value
from agents.orchestrator import build_orchestrator
from agents.session_bridge import build_backend_history, format_session_overview
from agents.source_view import format_citations, format_source_preview, format_sources_sidebar


APP_TITLE = "RAG Chatbot"
APP_SUBTITLE = "Khung chat nhóm về phòng chống ma túy - sẵn sàng để cắm RAG pipeline."
BANNER_PATH = "/public/chatbot_icon.png"

QUICK_PROMPTS = [
    "Những dấu hiệu nào thường được nhắc đến trong tài liệu về phòng chống ma túy?",
    "Nguồn nào trong tập tài liệu nói rõ biện pháp phòng chống nhất?",
    "Nếu không đủ evidence, chatbot nên trả lời thế nào để an toàn?",
]
DATA_MODE_KEY = "data_mode"


def get_data_mode() -> str:
    mode = get_value(DATA_MODE_KEY, "personal")
    return mode if mode in {"personal", "db"} else "personal"


async def send_chat_settings() -> None:
    await cl.ChatSettings(
        [
            Select(
                id=DATA_MODE_KEY,
                label="Data Mode",
                values=["personal", "db"],
                initial_index=0 if get_data_mode() == "personal" else 1,
                description="Chọn local corpus hoặc pgvector database.",
            )
        ]
    ).send()


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
        cl.Action(
            name="upload_documents",
            payload={},
            label="Upload docs",
            tooltip="Tải tài liệu lên mode hiện tại",
            icon="paperclip",
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
        "- Source previews hiển thị riêng.\n"
        "- Chọn được `personal` hoặc `db` mode.\n\n"
        "## Cách dùng nhanh\n"
        "1. Chọn data mode trong Chat Settings.\n"
        "2. Bấm `Upload docs` để ingest tài liệu vào mode hiện tại.\n"
        "3. Hỏi trực tiếp và xem citation + source previews ở panel bên phải."
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    if get_value("orchestrator") is None:
        set_value("orchestrator", build_orchestrator())
    orchestrator = get_value("orchestrator")
    set_value(DATA_MODE_KEY, get_data_mode())
    orchestrator.set_data_mode(get_data_mode())
    reset_history()
    await send_chat_settings()
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


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    mode = str(settings.get(DATA_MODE_KEY, "personal")).strip().lower()
    if mode not in {"personal", "db"}:
        mode = "personal"
    set_value(DATA_MODE_KEY, mode)
    orchestrator = get_value("orchestrator") or build_orchestrator()
    orchestrator.set_data_mode(mode)
    set_value("orchestrator", orchestrator)
    await cl.Message(
        content=f"Đã chuyển data mode sang `{mode}`.",
        author="RAG Assistant",
        tags=["settings"],
    ).send()


@cl.action_callback("upload_documents")
async def upload_documents_action(action: cl.Action) -> None:
    await action.remove()
    files = await cl.AskFileMessage(
        content="Tải tài liệu để ingest vào mode dữ liệu hiện tại.",
        accept=[
            "text/plain",
            "text/markdown",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ],
        max_files=5,
        max_size_mb=20,
        timeout=180,
    ).send()
    if not files:
        await cl.Message(content="Không có file nào được tải lên.", author="RAG Assistant").send()
        return

    orchestrator = get_value("orchestrator") or build_orchestrator()
    orchestrator.set_data_mode(get_data_mode())
    try:
        results = orchestrator.ingest_files(files)
    except Exception as exc:
        await cl.Message(
            content=f"Ingest thất bại: {exc}",
            author="RAG Assistant",
            actions=build_actions(),
            tags=["error", "ingest"],
        ).send()
        return
    summary = "\n".join(
        f"- {item.document_title} -> {item.chunk_count} chunks vào `{item.target_mode}`" for item in results
    )
    await cl.Message(
        content=f"Đã ingest {len(results)} tài liệu.\n\n{summary}",
        author="RAG Assistant",
        actions=build_actions(),
        tags=["ingest"],
    ).send()


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
    side_elements = [
        cl.Text(
            name="Source Summary",
            content=sidebar_text,
            display="side",
            language="markdown",
        )
    ]
    for idx, source in enumerate(response.sources[:4], start=1):
        preview_name, preview_content = format_source_preview(source, idx)
        side_elements.append(
            cl.Text(
                name=preview_name,
                content=preview_content,
                display="side",
                language="markdown",
            )
        )

    await cl.Message(
        content=answer,
        author="RAG Assistant",
        metadata=metadata,
        tags=["answer", "rag"],
        elements=side_elements,
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
    orchestrator.set_data_mode(get_data_mode())
    history = build_backend_history(get_history())
    append_turn("user", question)

    if is_quick_prompt:
        await cl.Message(
            content=f"**Câu hỏi mẫu**\n\n{question}",
            author="You",
            tags=["quick-prompt"],
        ).send()

    try:
        response = orchestrator.answer(question, history)
    except Exception as exc:
        await cl.Message(
            content=f"Không thể truy vấn nguồn dữ liệu hiện tại: {exc}",
            author="RAG Assistant",
            actions=build_actions(),
            tags=["error", "query"],
        ).send()
        return
    append_turn("assistant", response.answer)
    await render_response(response)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if not message.content.strip():
        await cl.Message(
            content="Hãy nhập câu hỏi hoặc dùng `Upload docs` để nạp thêm tài liệu.",
            author="RAG Assistant",
            actions=build_actions(),
            tags=["hint"],
        ).send()
        return
    await handle_question(message.content)
