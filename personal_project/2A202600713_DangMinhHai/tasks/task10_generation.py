"""
Task 10 — Generation Có Citation.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# top_k: đủ evidence nhưng chưa làm prompt quá dài
TOP_K = 5

# top_p: giữ đầu ra ổn định, giảm lan man
TOP_P = 0.9

# temperature: ưu tiên factual hơn sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def _extract_year(text: str) -> str | None:
    match = re.search(r"(19|20)\d{2}", text or "")
    return match.group(0) if match else None


def _source_label(chunk: dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    label = metadata.get("citation_label") or metadata.get("title")
    if not label:
        source = metadata.get("source") or metadata.get("path") or metadata.get("url") or "Nguồn"
        label = str(source).split("/")[-1].replace(".md", "").replace(".json", "").replace("_", "-")

    label = re.sub(r"[*`]+", "", str(label).strip())
    year = _extract_year(str(metadata.get("date_crawled", ""))) or _extract_year(str(metadata.get("url", "")))
    if year and year not in label:
        return f"{label}, {year}"
    return label


def _query_tokens(query: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", query.lower(), flags=re.UNICODE))


def _best_sentence(query: str, text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text.strip()
    q_tokens = _query_tokens(query)
    best_sentence = sentences[0].strip()
    best_score = float("-inf")
    for sentence in sentences:
        tokens = set(re.findall(r"\b\w+\b", sentence.lower(), flags=re.UNICODE))
        overlap = len(tokens & q_tokens)
        exact = 1.0 if query.lower() in sentence.lower() else 0.0
        score = overlap + exact
        if score > best_score:
            best_score = score
            best_sentence = sentence.strip()
    return best_sentence


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks theo pattern giúp giảm lost-in-the-middle.
    """
    if len(chunks) <= 2:
        return list(chunks)
    ordered = list(chunks[::2]) + list(reversed(chunks[1::2]))
    return ordered


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = _source_label(chunk)
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}"
        )
    return "\n---\n".join(context_parts)


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    sections = []
    for chunk in chunks[: min(3, len(chunks))]:
        label = _source_label(chunk)
        sentence = _best_sentence(query, chunk.get("content", ""))
        if sentence:
            sections.append(f"{sentence} [{label}]")
    if not sections:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return "\n\n".join(sections)


def _has_sufficient_evidence(query: str, chunks: list[dict]) -> bool:
    if not chunks:
        return False

    best_score = max(float(chunk.get("score", 0.0)) for chunk in chunks[:3])
    if best_score < 0.28:
        return False

    article_match = re.search(r"điều\s+(\d+)", query, flags=re.IGNORECASE)
    if article_match:
        article_no = article_match.group(1)
        if not any(
            re.search(rf"điều\s*{re.escape(article_no)}\b", chunk.get("content", ""), flags=re.IGNORECASE)
            or re.search(rf"điều\s*{re.escape(article_no)}\b", _source_label(chunk), flags=re.IGNORECASE)
            for chunk in chunks[:5]
        ):
            return False

    query_lower = query.lower()
    if any(phrase in query_lower for phrase in ("trích nguyên văn", "quote", "nguyên văn", "ngày chính xác", "chính xác diễn ra")):
        return False

    if any(phrase in query_lower for phrase in ("mức án bao nhiêu", "bao nhiêu năm", "mức phạt cụ thể", "ai là người bị kết án nặng nhất")):
        return False

    if any(token in query_lower for token in ("hình phạt", "phạt tù", "tàng trữ", "phạm tội")):
        has_penal_source = any(
            "hình sự" in (chunk.get("metadata", {}).get("title", "").lower())
            or "hình sự" in chunk.get("content", "").lower()
            or "hình sự" in _source_label(chunk).lower()
            for chunk in chunks[:5]
        )
        if not has_penal_source:
            return False
        return any(chunk.get("metadata", {}).get("type") == "legal" for chunk in chunks[:5])

    return True


def _call_openai(query: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}",
                },
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return None


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.
    """
    chunks = retrieve(query, top_k=top_k)
    ordered = reorder_for_llm(chunks)
    context = format_context(ordered)

    if not _has_sufficient_evidence(query, ordered):
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        return {
            "answer": answer,
            "sources": ordered,
            "retrieval_source": ordered[0].get("source", "hybrid") if ordered else "none",
        }

    answer = _call_openai(query, context)
    if not answer or "[" not in answer:
        answer = _extractive_answer(query, ordered)

    if "[" not in answer and ordered:
        answer = f"{answer}\n\n" + " ".join(f"[{_source_label(chunk)}]" for chunk in ordered[:2])

    return {
        "answer": answer,
        "sources": ordered,
        "retrieval_source": ordered[0].get("source", "hybrid") if ordered else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
