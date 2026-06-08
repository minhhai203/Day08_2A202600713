"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Logic pipeline:
    Query
      ├─→ Semantic Search (Task 5)  ─┐
      │                               ├─→ RRF Merge → Rerank (Task 7) → Results
      ├─→ Lexical Search (Task 6)  ──┘
      │
      └─→ Nếu reranker score < threshold
            └─→ Fallback: PageIndex (Task 8)

Dùng RRF để merge dense + sparse vì:
- Không cần normalize scores từ hai không gian khác nhau
- Robust với outlier scores
- k=60 (Cormack et al. 2009) được dùng rộng rãi trong hybrid search
"""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3   # Ngưỡng reranker score; dưới đây → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline với hybrid search + reranking + PageIndex fallback.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối
        score_threshold: Ngưỡng score tối thiểu (chỉ áp dụng khi dùng cross-encoder)
        use_reranking: Có dùng reranking không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'hybrid' | 'pageindex'
        }
    """
    # Step 1: Chạy dense + sparse song song (conceptually)
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    if not dense_results and not sparse_results:
        print("[WARN] Khong co ket qua -- dam bao da chay Task 4 de index du lieu")
        return []

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except Exception as e:
            print(f"  [WARN] Reranking that bai ({e}) -- dung RRF ket qua")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Step 4: Check threshold → fallback PageIndex
    # Chỉ so sánh threshold khi dùng cross-encoder (scores in [0,1])
    best_score = final_results[0]["score"] if final_results else 0.0
    should_fallback = (
        not final_results
        or (use_reranking and RERANK_METHOD == "cross_encoder" and best_score < score_threshold)
    )

    if should_fallback:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                print(f"  -> Fallback PageIndex (hybrid score={best_score:.3f} < {score_threshold})")
                return fallback
        except Exception as e:
            print(f"  [WARN] PageIndex fallback that bai: {e}")

    return final_results[:top_k]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r.get('source','?')}] {r['content'][:80]}...")
