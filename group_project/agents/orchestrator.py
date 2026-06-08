from __future__ import annotations

from typing import Any

from chainlit.types import AskFileResponse

from .contracts import ChatResponse
from .ingestion import ingest_db_files, ingest_personal_files
from .postgres_store import IngestResult
from .rag_manager import RAGManager, build_rag_manager


class ChatOrchestrator:
    """Thin orchestration layer so the team can split UI and RAG work cleanly."""

    def __init__(self, rag_manager: RAGManager | None = None, config: dict[str, Any] | None = None) -> None:
        self.rag_manager = rag_manager or build_rag_manager(config=config)

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> ChatResponse:
        return self.rag_manager.answer_chat(question=question, history=history or [])

    def set_data_mode(self, mode: str) -> None:
        self.rag_manager.set_data_mode(mode)

    def get_data_mode(self) -> str:
        return self.rag_manager.get_data_mode()

    def ingest_files(self, files: list[AskFileResponse]) -> list[IngestResult]:
        if self.get_data_mode() == "db":
            return ingest_db_files(files, self.rag_manager._postgres_store)  # lazy property, safe
        return ingest_personal_files(files)


def build_orchestrator(config: dict[str, Any] | None = None) -> ChatOrchestrator:
    return ChatOrchestrator(config=config)
