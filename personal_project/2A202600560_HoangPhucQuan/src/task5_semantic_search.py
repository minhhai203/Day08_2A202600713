"""
Task 5 — Semantic Search Module.

Dùng ChromaDB (đã index ở Task 4) để thực hiện dense retrieval.
ChromaDB cosine distance ∈ [0, 2] (0 = identical).
Chuyển sang similarity score: score = 1 - distance / 2 → [0, 1].
"""

from __future__ import annotations

try:
    from .task4_chunking_indexing import get_embedding_model, get_collection
except ImportError:
    from task4_chunking_indexing import get_embedding_model, get_collection  # type: ignore


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng cosine similarity trên ChromaDB.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    collection = get_collection()
    count = collection.count()
    if count == 0:
        print("[WARN] ChromaDB rong -- hay chay Task 4 truoc: python -m src.task4_chunking_indexing")
        return []

    n = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # cosine distance [0,2] → similarity [0,1]
        score = max(0.0, 1.0 - dist / 2.0)
        output.append({"content": doc, "score": float(score), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
