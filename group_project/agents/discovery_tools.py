from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parent.parent / "data" / "source_docs"

_SNIPPET_LEN = 300


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def get_source_root() -> Path:
    return SOURCE_ROOT


def list_available_sources() -> list[str]:
    root = get_source_root()
    if not root.exists():
        return []
    return sorted(
        _normalize(path.name)
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )


def list_sources_with_metadata() -> list[dict[str, Any]]:
    """Return rich metadata for each source document (title, source, url, snippet, type)."""
    from agents.corpus import load_documents  # local import to avoid circular deps

    results = []
    for doc in load_documents():
        meta = doc.get("metadata", {})
        content = doc.get("content", "")
        snippet = content[:_SNIPPET_LEN].strip()
        results.append(
            {
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "url": meta.get("source_url", ""),
                "snippet": snippet,
                "type": meta.get("type", "document"),
            }
        )
    return results


def read_source_text(filename: str) -> str:
    root = get_source_root()
    if not root.exists():
        return f"Chua co thu muc nguon: {root}"

    target = _normalize(filename)
    for path in root.rglob("*"):
        if path.is_file() and _normalize(path.name) == target:
            try:
                return path.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover
                return f"Khong doc duoc file {filename}: {exc}"
    return f"Khong tim thay file {filename}."

