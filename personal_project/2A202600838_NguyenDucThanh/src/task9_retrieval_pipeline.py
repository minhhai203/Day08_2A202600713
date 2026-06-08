"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback.

Logic:
    Query
      ├→ Semantic Search (Task 5)  ─┐
      │                              ├→ RRF Merge → Rerank (Task 7) → Results
      ├→ Lexical Search (Task 6)  ─┘
      │
      └→ Nếu best score < threshold → Fallback: PageIndex (Task 8)
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # Jina API nếu có key, fallback RRF nếu không


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu, dưới ngưỡng → dùng PageIndex
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Chạy semantic + lexical search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Normalize BM25 scores sang [0,1] để RRF công bằng hơn
    if sparse_results:
        max_score = max(r["score"] for r in sparse_results) or 1.0
        for r in sparse_results:
            r["score"] = r["score"] / max_score

    # Step 2: Merge kết quả bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: Kiểm tra threshold → fallback PageIndex
    # Dùng best dense score (cosine similarity) thay vì RRF/rerank score
    # để tránh false-negative khi Jina fallback sang RRF (score ~0.016)
    best_dense_score = dense_results[0]["score"] if dense_results else 0
    if not final_results or best_dense_score < score_threshold:
        score_val = best_dense_score
        print(f"  ⚠ Hybrid score ({score_val:.3f}) < threshold ({score_threshold}). Fallback → PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        return fallback if fallback else final_results

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r.get('source', '?')}] {r['content'][:80]}...")
