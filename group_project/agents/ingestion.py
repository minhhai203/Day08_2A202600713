from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chainlit.types import AskFileResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from .postgres_store import IngestResult, PostgresVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONAL_UPLOAD_ROOT = REPO_ROOT / "group_project" / "data" / "user_uploads" / "personal"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=160,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    source: str
    content: str
    url: str = ""
    metadata: dict[str, Any] | None = None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "uploaded-document"


def _first_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def parse_uploaded_file(file: AskFileResponse) -> ParsedDocument:
    path = Path(file["path"])
    filename = path.name
    fallback_title = path.stem.replace("-", " ").replace("_", " ").strip().title()

    if path.suffix.lower() in {".md", ".txt"}:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
    else:
        result = MarkItDown(enable_plugins=False).convert_local(path)
        content = (result.markdown or "").strip()

    if not content:
        raise RuntimeError(f"Không đọc được nội dung từ file {filename}.")

    title = _first_heading(content, fallback_title)
    return ParsedDocument(
        title=title,
        source=f"upload/{filename}",
        content=content,
        metadata={
            "uploaded_file": filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "mime_type": file.get("type", ""),
            "size": file.get("size", 0),
        },
    )


def chunk_document(content: str) -> list[str]:
    chunks = [chunk.strip() for chunk in splitter.split_text(content) if chunk.strip()]
    return chunks


def ingest_personal_files(files: list[AskFileResponse]) -> list[IngestResult]:
    PERSONAL_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[IngestResult] = []
    for file in files:
        document = parse_uploaded_file(file)
        slug = _slugify(document.title or Path(file["name"]).stem)
        output_path = PERSONAL_UPLOAD_ROOT / f"{slug}.md"
        output = (
            f"# {document.title}\n\n"
            f"**Source:** {document.source}\n"
            f"**Uploaded:** {document.metadata.get('uploaded_at', '') if document.metadata else ''}\n\n"
            f"{document.content.strip()}\n"
        )
        output_path.write_text(output, encoding="utf-8")
        results.append(
            IngestResult(
                document_id=slug,
                document_title=document.title,
                chunk_count=max(1, len(chunk_document(document.content))),
                target_mode="personal",
            )
        )
    return results


def ingest_db_files(files: list[AskFileResponse], store: PostgresVectorStore) -> list[IngestResult]:
    results: list[IngestResult] = []
    for file in files:
        document = parse_uploaded_file(file)
        chunks = chunk_document(document.content)
        results.append(
            store.ingest_document(
                title=document.title,
                source=document.source,
                content=document.content,
                chunks=chunks,
                metadata=document.metadata,
                url=document.url,
            )
        )
    return results
