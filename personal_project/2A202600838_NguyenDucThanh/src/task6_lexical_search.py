"""
Task 6 — Lexical Search Module (BM25).

BM25 (Best Match 25) hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao hơn
    - Inverse Document Frequency (IDF): từ hiếm trong corpus → quan trọng hơn
    - Length normalization: document dài không được ưu tiên quá mức
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Load corpus từ data/chunks.json (được tạo bởi Task 4) để BM25 và semantic
search hoạt động trên cùng một tập chunks.
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks.json"
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_corpus: list[dict] = []
_bm25: BM25Okapi = None


def _load_corpus() -> list[dict]:
    # Dùng chunks từ Task 4 để thống nhất với semantic search
    if CHUNKS_FILE.exists():
        return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    # Fallback nếu Task 4 chưa chạy: load full documents
    corpus = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        corpus.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xây dựng BM25 index từ corpus."""
    tokenized = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized)


def _ensure_index():
    global _corpus, _bm25
    if not _corpus:
        _corpus = _load_corpus()
        _bm25 = build_bm25_index(_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,   # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_index()

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": _corpus[idx]["metadata"]
            })

    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
