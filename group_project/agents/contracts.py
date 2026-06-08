from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Citation:
    id: str
    title: str
    source: str = ""
    url: str = ""


@dataclass(slots=True)
class SourceDocument:
    id: str
    title: str
    source: str = ""
    url: str = ""
    snippet: str = ""
    score: float | None = None


@dataclass(slots=True)
class ChatResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[SourceDocument] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [asdict(item) for item in self.citations],
            "sources": [asdict(item) for item in self.sources],
            "metadata": dict(self.metadata),
        }


DEFAULT_CHAT_RESPONSE = ChatResponse(
    answer="Khung chatbot đã sẵn sàng. Team có thể cắm pipeline RAG vào đây.",
    metadata={"used_memory": False, "retrieval_mode": "placeholder"},
)

