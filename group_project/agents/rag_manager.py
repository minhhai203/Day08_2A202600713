from __future__ import annotations

import os
import re
from textwrap import shorten
from typing import Any

from dotenv import load_dotenv

from .contracts import ChatResponse, Citation, SourceDocument
from .corpus import search_documents
from .postgres_store import PostgresVectorStore

load_dotenv()

TOP_K = int(os.getenv("GROUP_RAG_TOP_K", "5"))
TOP_P = float(os.getenv("GROUP_RAG_TOP_P", "0.9"))
TEMPERATURE = float(os.getenv("GROUP_RAG_TEMPERATURE", "0.25"))
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_DATA_MODE = os.getenv("GROUP_RAG_DATA_MODE", "personal")

SYSTEM_PROMPT = """Bạn là chatbot RAG cho chủ đề phòng chống ma túy và tin tức liên quan.
Chỉ sử dụng thông tin trong context được cung cấp.
Mọi câu trả lời phải trung thực, ngắn gọn, rõ ràng và có citation.
Nếu thông tin chưa đủ, hãy nói rõ là chưa thể xác minh từ nguồn hiện có.
Không bịa số liệu, không bịa sự kiện, không tự suy diễn quá mức.
Ưu tiên câu trả lời theo kiểu tư vấn an toàn, chính xác, hữu ích cho người đọc."""


class RAGManager:
    """Backend stub for Thành to wire vector retrieval and answer generation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._last_generation_mode = "extractive"
        self._postgres_store = PostgresVectorStore()

    def answer_chat(self, question: str, history: list[dict[str, str]] | None = None) -> ChatResponse:
        history = self._sanitize_history(history)
        query = self._build_query(question, history)
        sources = self.retrieve_sources(query, top_k=self.config.get("top_k", TOP_K))
        data_mode = self.get_data_mode()
        if not sources:
            return ChatResponse(
                answer="Tôi chưa tìm thấy nguồn phù hợp để xác minh câu hỏi này.",
                sources=[],
                metadata={
                    "used_memory": bool(history),
                    "retrieval_mode": f"{data_mode}-empty",
                    "generation_mode": "none",
                    "source_count": 0,
                    "data_mode": data_mode,
                },
            )

        answer = self._generate_answer(query, sources, history)
        generation_mode = "openai" if self._last_generation_mode == "openai" else "extractive"
        if not answer.strip():
            answer = self._extractive_answer(query, sources)
            generation_mode = "extractive"

        citations = self.build_citations(sources)
        answer = self._append_citation_footer(answer, citations)
        return ChatResponse(
            answer=answer.strip(),
            citations=citations,
            sources=sources,
            metadata={
                "used_memory": bool(history),
                "retrieval_mode": "local-corpus" if data_mode == "personal" else "pgvector",
                "generation_mode": generation_mode,
                "source_count": len(sources),
                "question_terms": len(re.findall(r"\b\w+\b", query.lower(), flags=re.UNICODE)),
                "data_mode": data_mode,
            },
        )

    def get_data_mode(self) -> str:
        mode = str(self.config.get("data_mode") or DEFAULT_DATA_MODE).strip().lower()
        return mode if mode in {"personal", "db"} else "personal"

    def set_data_mode(self, mode: str) -> None:
        self.config["data_mode"] = mode if mode in {"personal", "db"} else "personal"

    def retrieve_sources(
        self,
        question: str,
        top_k: int = TOP_K,
        history: list[dict[str, str]] | None = None,
    ) -> list[SourceDocument]:
        _ = history
        if self.get_data_mode() == "db":
            ranked = self._postgres_store.search(question, top_k=top_k)
        else:
            ranked = search_documents(question, top_k=top_k)
        return [self._to_source_document(item) for item in ranked]

    def build_citations(self, sources: list[SourceDocument]) -> list[Citation]:
        return [
            Citation(id=source.id, title=source.title, source=source.source, url=source.url)
            for source in sources
        ]

    def _sanitize_history(self, history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        clean_history: list[dict[str, str]] = []
        for turn in history or []:
            role = (turn.get("role") or "").strip()
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                clean_history.append({"role": role, "content": content})
        return clean_history

    def _build_query(self, question: str, history: list[dict[str, str]]) -> str:
        question = question.strip()
        if not history:
            return question

        followup_markers = ("nó", "vậy", "đó", "cái đó", "cái này", "phần đó", "tiếp", "thêm")
        if len(question) < 18 or question.lower().startswith(followup_markers) or any(
            marker in question.lower() for marker in followup_markers
        ):
            prior_user = [turn["content"] for turn in history if turn["role"] == "user"]
            if prior_user:
                return f"{question}\n\nNgữ cảnh trước đó: {prior_user[-1]}"
        return question

    def _to_source_document(self, item: dict[str, Any]) -> SourceDocument:
        metadata = item.get("metadata", {}) or {}
        source_id = metadata.get("source") or metadata.get("path") or metadata.get("title") or "source"
        return SourceDocument(
            id=str(source_id),
            title=str(metadata.get("title") or metadata.get("citation_label") or source_id),
            source=str(metadata.get("source") or metadata.get("path") or ""),
            url=str(metadata.get("source_url") or metadata.get("url") or ""),
            snippet=shorten(item.get("content", "").replace("\n", " "), width=240, placeholder="..."),
            score=float(item.get("score", 0.0)) if item.get("score") is not None else None,
        )

    def _source_label(self, source: SourceDocument) -> str:
        return source.title or source.id

    def _append_citation_footer(self, answer: str, citations: list[Citation]) -> str:
        answer = answer.strip()
        if not citations:
            return answer
        footer = "Nguồn: " + " | ".join(f"[{idx}] {citation.title}" for idx, citation in enumerate(citations, start=1))
        return f"{answer}\n\n{footer}"

    def _format_context(self, sources: list[SourceDocument]) -> str:
        blocks = []
        for idx, source in enumerate(sources, 1):
            header = f"[Document {idx} | Source: {source.title}]"
            if source.source:
                header = f"{header} ({source.source})"
            blocks.append(f"{header}\n{source.snippet or ''}")
        return "\n---\n".join(blocks)

    def _call_openai(self, query: str, context: str) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self._last_generation_mode = "extractive"
            return None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", OPENAI_CHAT_MODEL),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            self._last_generation_mode = "openai"
            return response.choices[0].message.content or ""
        except Exception:
            self._last_generation_mode = "extractive"
            return None

    def _extractive_answer(self, query: str, sources: list[SourceDocument]) -> str:
        if not sources:
            return "Tôi chưa tìm thấy nguồn phù hợp để xác minh câu hỏi này."

        query_tokens = set(re.findall(r"\b\w+\b", query.lower(), flags=re.UNICODE))
        sections = []
        for source in sources[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", source.snippet.strip()) if source.snippet else []
            best_sentence = ""
            best_score = float("-inf")
            for sentence in sentences or [source.snippet]:
                tokens = set(re.findall(r"\b\w+\b", sentence.lower(), flags=re.UNICODE))
                overlap = len(tokens & query_tokens)
                exact = 1.0 if query.lower() in sentence.lower() else 0.0
                score = overlap + exact
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
            if best_sentence:
                sections.append(f"{best_sentence} [{source.title}]")

        return "\n\n".join(sections) if sections else "Tôi chưa tìm thấy nguồn phù hợp để xác minh câu hỏi này."

    def _generate_answer(self, query: str, sources: list[SourceDocument], history: list[dict[str, str]]) -> str:
        context = self._format_context(sources)
        answer = self._call_openai(query, context)
        if answer:
            return answer
        self._last_generation_mode = "extractive"
        answer = self._extractive_answer(query, sources)
        if history and answer:
            return answer
        return answer


def build_rag_manager(config: dict[str, Any] | None = None) -> RAGManager:
    return RAGManager(config=config)
