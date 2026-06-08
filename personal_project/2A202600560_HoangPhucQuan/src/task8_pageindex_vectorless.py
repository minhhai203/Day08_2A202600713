"""
Task 8 — PageIndex Vectorless RAG.

PageIndex thực hiện RAG mà không cần vector embedding — dùng structural
understanding của document (tree/page layout) để retrieve.

API flow (pageindex SDK v0.2.x):
  1. submit_document(file_path) → {'doc_id': ...}  (PDF only)
  2. is_retrieval_ready(doc_id) → bool  (poll until True)
  3. submit_query(doc_id, query) → {'retrieval_id': ...}
  4. get_retrieval(retrieval_id) → {'status': ..., ...}  (poll until done)

Setup:
  - Đăng ký tại https://pageindex.ai/ → lấy API key
  - Thêm PAGEINDEX_API_KEY vào .env
  - Chạy upload_documents() một lần để index PDF files

Lưu ý: PageIndex chỉ hỗ trợ PDF (không hỗ trợ markdown/DOCX).
Upload từ data/landing/legal/*.pdf
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

MAX_POLL_SECONDS = 60   # tối đa chờ 60 giây cho mỗi query


def _get_client():
    """Khởi tạo PageIndexClient. Raise nếu API key chưa set."""
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY in ("pi_xxx", ""):
        raise ValueError(
            "PAGEINDEX_API_KEY not set.\n"
            "  1. Register at https://pageindex.ai/\n"
            "  2. Add PAGEINDEX_API_KEY=<key> to .env"
        )
    from pageindex import PageIndexClient  # type: ignore
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_ids() -> dict[str, str]:
    """Load saved doc_ids từ file JSON {filename: doc_id}."""
    if DOC_IDS_FILE.exists():
        return json.loads(DOC_IDS_FILE.read_text())
    return {}


def _save_doc_ids(doc_ids: dict[str, str]) -> None:
    DOC_IDS_FILE.write_text(json.dumps(doc_ids, indent=2))


def upload_documents() -> dict[str, str]:
    """
    Upload các PDF files lên PageIndex.
    Bỏ qua files đã upload (dựa theo tên file trong cache).
    Trả về {filename: doc_id}.
    """
    pi = _get_client()
    doc_ids = _load_doc_ids()

    pdf_files = list(LANDING_LEGAL_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[WARN] Khong tim thay PDF files trong data/landing/legal/")
        return doc_ids

    for pdf_file in pdf_files:
        if pdf_file.name in doc_ids:
            print(f"  [SKIP] Already uploaded: {pdf_file.name} (doc_id={doc_ids[pdf_file.name]})")
            continue

        result = pi.submit_document(file_path=str(pdf_file))
        doc_id = result.get("doc_id", "")
        if doc_id:
            doc_ids[pdf_file.name] = doc_id
            print(f"  [OK] Uploaded: {pdf_file.name} -> doc_id={doc_id}")
        else:
            print(f"  [WARN] Upload failed for {pdf_file.name}: {result}")

    _save_doc_ids(doc_ids)
    print(f"\n[OK] {len(doc_ids)} documents indexed in PageIndex")
    return doc_ids


def _poll_retrieval(pi, retrieval_id: str, timeout: int = MAX_POLL_SECONDS) -> dict:
    """Poll get_retrieval cho đến khi status=completed hoặc timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = pi.get_retrieval(retrieval_id)
        status = result.get("status", "")
        if status in ("completed", "done", "success"):
            return result
        if status in ("failed", "error"):
            raise RuntimeError(f"PageIndex retrieval failed: {result}")
        time.sleep(2)
    raise TimeoutError(f"PageIndex retrieval timed out after {timeout}s")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval bằng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả đủ tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    pi = _get_client()
    doc_ids = _load_doc_ids()

    if not doc_ids:
        raise RuntimeError(
            "No documents uploaded. Run upload_documents() first: "
            "python -m src.task8_pageindex_vectorless"
        )

    all_results: list[dict] = []

    for filename, doc_id in doc_ids.items():
        # Check if document is ready
        if not pi.is_retrieval_ready(doc_id):
            print(f"  [WARN] Document not ready: {filename} (still indexing)")
            continue

        # Submit query
        r = pi.submit_query(doc_id=doc_id, query=query)
        retrieval_id = r.get("retrieval_id", "")
        if not retrieval_id:
            print(f"  [WARN] No retrieval_id for {filename}: {r}")
            continue

        # Poll for results
        result = _poll_retrieval(pi, retrieval_id)

        # Parse retrieved_nodes (API v0.2.x structure)
        # node = {'id', 'title', 'metadata', 'relevant_contents': [[{'relevant_content': ...}]]}
        nodes = result.get("retrieved_nodes") or []
        for rank, node in enumerate(nodes):
            # Flatten all relevant_content from nested list
            content_parts: list[str] = []
            for group in node.get("relevant_contents", []):
                for item in group:
                    part = item.get("relevant_content", "")
                    if part:
                        content_parts.append(part)
            content = "\n\n".join(content_parts) if content_parts else node.get("title", "")
            score = 1.0 / (1 + rank)   # rank-based score: 1st=1.0, 2nd=0.5, ...
            all_results.append({
                "content": content,
                "score": score,
                "metadata": {
                    "source": filename,
                    "type": "legal",
                    "doc_id": doc_id,
                    "node_id": node.get("id", ""),
                    "title": node.get("title", ""),
                },
                "source": "pageindex",
            })

    # Sort by score, return top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY in ("pi_xxx", ""):
        print("[WARN] PAGEINDEX_API_KEY chua duoc set trong .env")
        print("  Dang ky tai: https://pageindex.ai/")
    else:
        print("Uploading documents to PageIndex...")
        upload_documents()

        print("\nWaiting for indexing (may take 30-60 seconds for first upload)...")
        time.sleep(5)

        print("\nTest query:")
        results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
