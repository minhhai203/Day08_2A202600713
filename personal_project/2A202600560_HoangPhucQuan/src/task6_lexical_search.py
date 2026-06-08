"""
Task 6 — Lexical Search Module (BM25).

BM25 (Okapi BM25) hoạt động:
- TF: từ xuất hiện nhiều trong document → điểm cao (có saturation)
- IDF: từ hiếm trong corpus → quan trọng hơn từ phổ biến
- Length norm: document dài không được ưu tiên quá mức (tham số b=0.75)
- Formula: score(q,d) = Σ IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
  k1=1.5, b=0.75 (Okapi BM25 defaults)

Corpus được load từ ChromaDB (đã index ở Task 4) để đồng nhất dữ liệu.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from .task4_chunking_indexing import get_collection
except ImportError:
    from task4_chunking_indexing import get_collection  # type: ignore

_bm25: BM25Okapi | None = None
_corpus: list[dict] = []


def _load_corpus_from_chroma() -> list[dict]:
    """Load toàn bộ chunks từ ChromaDB vào memory."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.get(
        limit=count,
        include=["documents", "metadatas"],
    )
    return [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


def build_bm25_index(corpus: list[dict] | None = None) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Tokenize bằng .lower().split() — đủ cho tiếng Việt (ký tự cách phân biệt từ).
    """
    global _bm25, _corpus
    if corpus is None:
        corpus = _load_corpus_from_chroma()
    _corpus = corpus
    tokenized = [doc["content"].lower().split() for doc in corpus]
    _bm25 = BM25Okapi(tokenized)
    return _bm25


def _get_bm25() -> BM25Okapi:
    global _bm25
    if _bm25 is None:
        build_bm25_index()
    if _bm25 is None:
        raise RuntimeError("Corpus rong -- hay chay Task 4 truoc: python -m src.task4_chunking_indexing")
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng BM25Okapi.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    global _corpus

    bm25 = _get_bm25()
    if not _corpus:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": _corpus[idx]["metadata"],
            })

    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
