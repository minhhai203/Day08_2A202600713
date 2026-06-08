from __future__ import annotations

from typing import Any

from .contracts import ChatResponse, Citation, DEFAULT_CHAT_RESPONSE, SourceDocument


class RAGManager:
    """Backend stub for Thành to wire vector retrieval and answer generation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def answer_chat(self, question: str, history: list[dict[str, str]] | None = None) -> ChatResponse:
        _ = question, history
        return DEFAULT_CHAT_RESPONSE

    def retrieve_sources(self, question: str) -> list[SourceDocument]:
        _ = question
        return []

    def build_citations(self, sources: list[SourceDocument]) -> list[Citation]:
        return [
            Citation(id=source.id, title=source.title, source=source.source, url=source.url)
            for source in sources
        ]


def build_rag_manager(config: dict[str, Any] | None = None) -> RAGManager:
    return RAGManager(config=config)

