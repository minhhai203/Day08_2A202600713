"""
Task 5 — Semantic Search Module.

Dense retrieval trên ChromaDB sử dụng Google text-embedding-004.
Phải chạy Task 4 trước để build index.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-2"
CHROMA_DIR = str(Path(__file__).parent.parent / "data" / "chroma_db")
COLLECTION_NAME = "drug_law_docs"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _embed_query(query: str) -> list[float]:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,   # cosine similarity [0, 1]
            'metadata': dict
        }
        Sorted by score descending.
    """
    query_embedding = _embed_query(query)
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # ChromaDB cosine distance = 1 - cosine_similarity
        score = 1.0 - dist
        output.append({
            "content": doc,
            "score": float(score),
            "metadata": meta
        })

    return sorted(output, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
