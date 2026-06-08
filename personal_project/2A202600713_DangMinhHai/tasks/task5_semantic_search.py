"""
Task 5 — Semantic Search Module.
"""

from __future__ import annotations

from .task4_chunking_indexing import semantic_search as _semantic_search


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.
    """
    return _semantic_search(query, top_k=top_k)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
