from __future__ import annotations

from typing import Any

from .contracts import ChatResponse
from .rag_manager import RAGManager, build_rag_manager


class ChatOrchestrator:
    """Thin orchestration layer so the team can split UI and RAG work cleanly."""

    def __init__(self, rag_manager: RAGManager | None = None, config: dict[str, Any] | None = None) -> None:
        self.rag_manager = rag_manager or build_rag_manager(config=config)

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> ChatResponse:
        return self.rag_manager.answer_chat(question=question, history=history or [])


def build_orchestrator(config: dict[str, Any] | None = None) -> ChatOrchestrator:
    return ChatOrchestrator(config=config)

