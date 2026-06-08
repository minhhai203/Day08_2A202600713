"""
Task 6 — Lexical Search Module.

Mặc định vẫn dùng BM25 để giữ đúng yêu cầu gốc của bài.
Bonus: thêm TF-IDF + cosine similarity để demo phương pháp lexical khác.

Giải thích ngắn cho bonus:
- TF-IDF gán trọng số cao cho từ xuất hiện nhiều trong document nhưng hiếm trong toàn corpus.
- Sau đó dùng cosine similarity giữa vector query và vector document để xếp hạng.
- So với BM25, TF-IDF đơn giản hơn, dễ giải thích, và đủ tốt cho corpus nhỏ/khóa học.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .task4_chunking_indexing import _get_index, lexical_search as _bm25_search, load_documents

LEXICAL_METHOD = "bm25"  # "bm25" | "tfidf"


def _corpus_items() -> list[dict]:
    # Ưu tiên chunks đã index; fallback về documents nếu index chưa sẵn sàng.
    corpus = _get_index()
    if corpus:
        return corpus
    docs = load_documents()
    return [
        {
            "content": doc["content"],
            "metadata": doc.get("metadata", {}),
        }
        for doc in docs
    ]


@lru_cache(maxsize=1)
def _build_tfidf_vectorizer() -> tuple[TfidfVectorizer, np.ndarray, list[dict]]:
    corpus = _corpus_items()
    texts = [item.get("search_text") or item["content"] for item in corpus]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, corpus


def lexical_search_tfidf(query: str, top_k: int = 10) -> list[dict]:
    """
    Lexical search bằng TF-IDF + cosine similarity.

    Trả về cùng schema với BM25:
        {'content': str, 'score': float, 'metadata': dict}
    """
    corpus = _corpus_items()
    if not corpus:
        return []

    vectorizer, matrix, _ = _build_tfidf_vectorizer()
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).ravel()

    results = []
    order = np.argsort(scores)[::-1]
    for idx in order[:top_k]:
        score = float(scores[idx])
        if score <= 0 and results:
            continue
        item = corpus[idx]
        results.append(
            {
                "content": item["content"],
                "score": score,
                "metadata": item.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def lexical_search(query: str, top_k: int = 10, method: str = LEXICAL_METHOD) -> list[dict]:
    """
    Lexical search chính thức của bài.

    method = "bm25"  -> dùng BM25 như yêu cầu gốc
    method = "tfidf" -> bonus path để demo phương pháp lexical khác
    """
    if method == "tfidf":
        return lexical_search_tfidf(query, top_k=top_k)
    return _bm25_search(query, top_k=top_k)


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    print("BM25:")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

    print("\nTF-IDF bonus:")
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5, method="tfidf")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
