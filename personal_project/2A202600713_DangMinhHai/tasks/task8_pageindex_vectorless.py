"""
Task 8 — PageIndex Vectorless RAG.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import load_documents

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_DIR = Path(__file__).parent.parent / "data" / ".cache"
PAGEINDEX_CACHE_PATH = CACHE_DIR / "task8_pageindex.json"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE))


def _doc_label(metadata: dict) -> str:
    return metadata.get("source") or metadata.get("path") or "document"


def _score_document(query: str, document: dict) -> float:
    metadata = document.get("metadata", {}) or {}
    content = document.get("search_text") or document.get("content", "")
    content_tokens = _tokenize(content)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    exact_phrase = 1.0 if query.lower() in content.lower() else 0.0
    heading_boost = 0.0
    first_line = metadata.get("citation_label") or next((line.strip("# ").strip() for line in content.splitlines() if line.strip()), "")
    if first_line and any(token in first_line.lower() for token in query_tokens):
        heading_boost = 0.25

    return float(0.6 * overlap + 0.25 * exact_phrase + 0.15 * heading_boost)


def _load_cache() -> list[dict]:
    if PAGEINDEX_CACHE_PATH.exists():
        try:
            payload = json.loads(PAGEINDEX_CACHE_PATH.read_text(encoding="utf-8"))
            return payload.get("documents", [])
        except Exception:
            return []
    return []


def _save_cache(documents: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_CACHE_PATH.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False),
        encoding="utf-8",
    )


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    documents = load_documents()
    if not documents:
        return []

    if not PAGEINDEX_API_KEY:
        _save_cache(documents)
        return documents

    try:
        from pageindex import PageIndexClient

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        uploaded = []
        for doc in documents:
            path = doc.get("metadata", {}).get("path")
            if not path:
                continue
            response = client.submit_document(file_path=path)
            uploaded.append(
                {
                    **doc,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "pageindex_doc_id": response.get("doc_id") or response.get("id"),
                    },
                }
            )
        if uploaded:
            _save_cache(uploaded)
        return uploaded
    except Exception:
        _save_cache(documents)
        return documents


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    """
    documents = _load_cache()
    if not documents:
        documents = load_documents()
        if not documents:
            return []

    scored = []
    for doc in documents:
        score = _score_document(query, doc)
        scored.append(
            {
                "content": doc["content"],
                "score": score,
                "metadata": {
                    **doc.get("metadata", {}),
                    "citation_label": doc.get("metadata", {}).get("citation_label")
                    or doc.get("metadata", {}).get("title")
                    or _doc_label(doc.get("metadata", {})),
                },
                "source": "pageindex",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
