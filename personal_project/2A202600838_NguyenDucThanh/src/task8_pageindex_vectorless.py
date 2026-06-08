"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG không cần vector store — dùng structural understanding
của PDF document thay vì embedding.

Flow:
    1. upload_documents(): upload PDF files → lưu doc_ids vào data/pageindex_docs.json
    2. pageindex_search(): submit query → poll cho tới khi có kết quả → trả về chunks

Cài đặt:
    pip install pageindex

Yêu cầu: điền PAGEINDEX_API_KEY vào .env
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"


def _get_client():
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict:
    """
    Upload các file PDF từ data/landing/legal/ lên PageIndex.
    Lưu doc_ids vào data/pageindex_docs.json để tái sử dụng.

    Returns:
        dict: {'filename': 'doc_id', ...}
    """
    if not PAGEINDEX_API_KEY:
        print("PageIndex: PAGEINDEX_API_KEY not set in .env")
        return {}

    client = _get_client()
    doc_ids = {}

    for pdf_file in LANDING_LEGAL_DIR.glob("*.pdf"):
        print(f"  Uploading: {pdf_file.name}")
        result = client.submit_document(file_path=str(pdf_file))
        doc_id = result["doc_id"]
        doc_ids[pdf_file.name] = doc_id
        print(f"  [OK] doc_id: {doc_id}")

    DOC_IDS_FILE.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved doc_ids to {DOC_IDS_FILE}")
    return doc_ids


def _load_doc_ids() -> dict:
    """Load doc_ids đã upload trước đó."""
    if DOC_IDS_FILE.exists():
        return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    return {}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex chat-completions API.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Flow: chat_completions(doc_id, query, enable_citations=True) → parse response

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if not PAGEINDEX_API_KEY:
        print("  PageIndex: PAGEINDEX_API_KEY not set — skipping fallback")
        return []

    doc_ids = _load_doc_ids()
    if not doc_ids:
        print("  PageIndex: no documents uploaded — run upload_documents() first")
        return []

    client = _get_client()
    results = []

    for filename, doc_id in doc_ids.items():
        try:
            resp = client.chat_completions(
                messages=[{"role": "user", "content": query}],
                doc_id=doc_id,
                enable_citations=True,
            )
            content = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not content:
                content = str(resp)

            # Trả về response như 1 chunk duy nhất với rank-based score
            results.append({
                "content": content,
                "score": 1.0,
                "metadata": {"source": filename, "type": "legal"},
                "source": "pageindex"
            })
        except Exception as e:
            print(f"  PageIndex error for {filename}: {e}")

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Hay set PAGEINDEX_API_KEY trong file .env")
        print("Dang ky tai: https://pageindex.ai/")
    else:
        doc_ids = _load_doc_ids()
        if not doc_ids:
            print("Uploading documents...")
            upload_documents()

        print("\nTest query:")
        results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
