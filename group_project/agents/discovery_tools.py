from __future__ import annotations

import unicodedata
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent.parent / "data" / "source_docs"


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

