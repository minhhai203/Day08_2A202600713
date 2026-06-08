"""
Task 10 — Generation Có Citation.

LLM: Gemini 1.5 Flash (google-generativeai)
    Cần điền GEMINI_API_KEY vào .env

Tham số generation:
    - temperature=0.3: ưu tiên factual accuracy, ít sáng tạo — phù hợp RAG pháp luật
    - top_p=0.9: đủ diverse nhưng không quá random

Document reordering (tránh lost-in-the-middle):
    LLM nhớ tốt thông tin ở đầu và cuối, kém nhớ ở giữa.
    Strategy: chunks quan trọng nhất → đầu và cuối, kém nhất → giữa.
    Ví dụ [1,2,3,4,5] → [1,3,5,4,2]
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Trả lời câu hỏi sau bằng tiếng Việt một cách đầy đủ và rõ ràng.

Mỗi thông tin hay luận điểm đưa ra PHẢI kèm theo trích dẫn nguồn ngay sau đó,
sử dụng đúng tên file nguồn và năm từ context được cung cấp.

Định dạng trích dẫn: [<tên_file>, <năm>]
Ví dụ:
  - [120_2025_QH15_666019.md, 2025]
  - [huong-dan-luat-phong-chong-ma-tuy.md, 2021]
  - [bai-bao-nghe-si-x.md, 2024]

Nếu không xác định được năm, dùng năm gần nhất có thể suy ra từ tên file hoặc nội dung.

Cuối câu trả lời, liệt kê mục "**Tài liệu tham khảo:**" gồm tất cả các file nguồn đã dùng.

Nếu thông tin không có trong context được cung cấp, hãy trả lời:
'I cannot verify this information'

Quy tắc:
- Chỉ dùng thông tin từ context được cung cấp, không bịa đặt
- Mọi luận điểm đều PHẢI có trích dẫn [Nguồn, Năm]
- Nếu context không đủ, nói rõ điều đó"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Tránh lost-in-the-middle: đặt chunks quan trọng ở đầu và cuối.

    Với n chunks sorted theo score desc [1,2,3,4,5]:
    - Vị trí chẵn (0,2,4) → front: [1,3,5]
    - Vị trí lẻ (1,3) → back reversed: [4,2]
    - Kết quả: [1,3,5,4,2]
    """
    if len(chunks) <= 2:
        return chunks

    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)][::-1]
    return front + back


def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string với source labels để LLM có thể cite."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {
            "answer": "⚠ GEMINI_API_KEY not set in .env — không thể generate",
            "sources": [],
            "retrieval_source": "none"
        }

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với source labels
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call Gemini (retry tối đa 3 lần nếu gặp 503 overload)
    import time as _time
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=user_message,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    system_instruction=SYSTEM_PROMPT,
                )
            )
            break
        except Exception as e:
            if attempt < 2 and "503" in str(e):
                print(f"  Gemini 503, retry {attempt + 1}/3...")
                _time.sleep(5)
            else:
                raise

    answer = response.text

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
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
