"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve chunks (Task 9)
    2. Reorder để tránh "lost in the middle" (Liu et al. 2023)
    3. Format context với source labels để LLM có thể cite
    4. Build prompt với SYSTEM_PROMPT
    5. Gọi OpenAI GPT-4o-mini
    6. Return answer có citation + sources

Tham số LLM:
- temperature=0.3: RAG cần factual, ít sáng tạo — nhiệt độ thấp giảm hallucination
- top_p=0.9: vẫn đủ diverse cho câu trả lời dài, không quá rigid
- model=gpt-4o-mini: nhanh, rẻ, đủ tốt cho task grounding
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TOP_K = 5           # Đủ evidence mà không quá dài gây lost in the middle
TOP_P = 0.9         # Nucleus sampling: diverse nhưng có kiểm soát
TEMPERATURE = 0.3   # Factual output, hạn chế hallucination
LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source document number provided in the context
(e.g., [Document 1], [Document 3, Điều 248]).

Rules:
- ONLY use information from the provided context documents
- Every factual claim MUST have a citation like [Document N]
- If the context is insufficient to answer, say exactly:
  "Tôi không thể xác minh thông tin này từ nguồn hiện có."
- Structure your answer with clear paragraphs
- Answer in Vietnamese"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" (Liu et al. 2023).

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI, quên thông tin ở GIỮA.
    Strategy: chunks quan trọng nhất (score cao) ở đầu và cuối.

    Ví dụ với 5 chunks (sorted best→worst: 0,1,2,3,4):
      odds  = [0, 2, 4]  → đặt ở đầu
      evens = [1, 3]     → đảo ngược → [3, 1] → đặt ở cuối
      result = [0, 2, 4, 3, 1]

    Map 1-indexed: [1, 3, 5, 4, 2] — khớp ví dụ trong README.
    """
    if len(chunks) <= 2:
        return chunks

    odds = chunks[::2]          # indices 0,2,4... (best, 3rd, 5th)
    evens = chunks[1::2]        # indices 1,3,5... (2nd, 4th)
    return odds + evens[::-1]   # second-best ở cuối → LLM chú ý nhiều hơn


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label [Document N] để LLM cite đúng nguồn.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"source_{i}")
        doc_type = meta.get("type", "unknown")
        score = chunk.get("score", 0.0)
        parts.append(
            f"[Document {i} | File: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation với citation.

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng (sau reorder)
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY chưa được set trong .env")

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder (tránh lost in the middle)
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = (
        f"Context:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )

    # Step 5: Call LLM
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content or ""

    # Step 6: Return
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy theo phap luat Viet Nam?",
        "Nhung nghe si nao da bi bat vi lien quan toi ma tuy?",
        "Quy trinh cai nghien bat buoc theo Luat Phong chong ma tuy 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
