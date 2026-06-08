from __future__ import annotations

import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = [
    REPO_ROOT / "group_project" / "data" / "source_docs",
    REPO_ROOT / "personal_project" / "2A202600713_DangMinhHai" / "data" / "standardized",
]

LEGAL_TITLE_MAP = {
    "luat-phong-chong-ma-tuy-2021": "Luật Phòng, chống ma tuý 2021",
    "nghi-dinh-105-2021-nd-cp": "Nghị định 105/2021/NĐ-CP",
    "luat-120-2025": "Luật 120/2025",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip().lstrip("# ").strip()
        if candidate:
            return candidate
    return ""


def _parse_markdown_metadata(content: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    lines = content.splitlines()[:24]
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**Source:**"):
            metadata["source_url"] = stripped.split("**Source:**", 1)[-1].strip()
        elif stripped.startswith("**Crawled:**"):
            metadata["date_crawled"] = stripped.split("**Crawled:**", 1)[-1].strip()
        elif stripped.startswith("# ") and "title" not in metadata:
            metadata["title"] = stripped.lstrip("# ").strip()
    return metadata


def _doc_type_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "document"


def _source_label(path: Path) -> str:
    for root in SOURCE_ROOTS:
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except Exception:
            continue
    return path.name


def _source_root_label(path: Path) -> str:
    for root in SOURCE_ROOTS:
        try:
            path.relative_to(root)
            return str(root)
        except Exception:
            continue
    return str(path.parent)


def iter_source_files() -> list[Path]:
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        files = sorted(root.rglob("*.md"))
        if not files:
            continue
        return files
    return []


def load_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for filepath in iter_source_files():
        content = _normalize_whitespace(filepath.read_text(encoding="utf-8"))
        path_meta = _parse_markdown_metadata(content)
        doc_type = _doc_type_from_path(filepath)
        title = path_meta.get("title")
        if doc_type == "legal":
            title = LEGAL_TITLE_MAP.get(filepath.stem, title or filepath.stem.replace("-", " ").title())
        if not title or title.strip() in {"|  |  |", "||"}:
            title = _first_heading(content)
        if not title or title.strip() in {"|  |  |", "||"}:
            title = filepath.stem.replace("-", " ").title()
        citation_label = title
        if doc_type == "news" and path_meta.get("date_crawled"):
            year_match = re.search(r"(19|20)\d{2}", path_meta["date_crawled"])
            if year_match and year_match.group(0) not in citation_label:
                citation_label = f"{citation_label}, {year_match.group(0)}"

        documents.append(
            {
                "content": content,
                "search_text": f"{title}\n{content}",
                "metadata": {
                    "title": title,
                    "citation_label": citation_label,
                    "source": _source_label(filepath),
                    "source_root": _source_root_label(filepath),
                    "type": doc_type,
                    "path": str(filepath),
                    **path_meta,
                },
            }
        )
    return documents


def score_document(query: str, document: dict[str, Any]) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    metadata = document.get("metadata", {}) or {}
    content = document.get("search_text") or document.get("content", "")
    content_tokens = _tokenize(content)
    title_tokens = _tokenize(metadata.get("title", ""))

    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
    exact_phrase = 1.0 if query.lower() in content.lower() else 0.0
    heading_boost = 0.25 if any(token in metadata.get("title", "").lower() for token in query_tokens) else 0.0

    return float(0.50 * overlap + 0.25 * title_overlap + 0.20 * exact_phrase + 0.05 * heading_boost)


def search_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    documents = load_documents()
    if not documents:
        return []

    scored = []
    for doc in documents:
        scored.append(
            {
                "content": doc["content"],
                "score": score_document(query, doc),
                "metadata": dict(doc.get("metadata", {})),
                "source": "local-corpus",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
