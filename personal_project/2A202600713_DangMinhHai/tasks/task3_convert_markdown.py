"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft cho PDF/DOCX, và chuẩn hóa JSON crawl
thành markdown có metadata header.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _write_markdown(source: Path, output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  ✓ Saved: {output}")


def _fallback_pdf_markdown(filepath: Path) -> str:
    """Fallback cho PDF scan không có text extractable."""
    pdfinfo = ""
    try:
        result = subprocess.run(
            ["pdfinfo", str(filepath)],
            capture_output=True,
            text=True,
            check=False,
        )
        pdfinfo = result.stdout.strip()
    except Exception:
        pdfinfo = ""

    info_lines = [
        f"# {filepath.stem}",
        "",
        "Tệp PDF này không trích xuất được văn bản máy đọc từ MarkItDown.",
        "Nội dung dưới đây là phần mô tả thay thế để giữ pipeline Markdown hoạt động.",
        "",
        f"- File gốc: {filepath.name}",
        f"- Kích thước: {filepath.stat().st_size} bytes",
    ]

    if pdfinfo:
        pages_line = next((line for line in pdfinfo.splitlines() if line.startswith("Pages:")), "")
        creator_line = next((line for line in pdfinfo.splitlines() if line.startswith("Creator:")), "")
        producer_line = next((line for line in pdfinfo.splitlines() if line.startswith("Producer:")), "")
        if pages_line:
            info_lines.append(f"- {pages_line}")
        if creator_line:
            info_lines.append(f"- {creator_line}")
        if producer_line:
            info_lines.append(f"- {producer_line}")

    info_lines.extend(
        [
            "",
            "Ghi chú:",
            "Tài liệu pháp luật này vẫn được giữ trong `data/landing/legal/` để phục vụ bài lab.",
            "Nếu cần nội dung chi tiết hơn cho demo, có thể thay PDF scan bằng bản text/PDF gốc từ nguồn chính thống.",
        ]
    )
    return "\n".join(info_lines)


def convert_legal_docs() -> None:
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        result = md.convert(str(filepath))
        text = result.text_content or ""
        if filepath.suffix.lower() == ".pdf" and len(text.strip()) < 200:
            text = _fallback_pdf_markdown(filepath)
        else:
            text = result.text_content
        output_path = output_dir / f"{filepath.stem}.md"
        _write_markdown(filepath, output_path, text)


def convert_news_articles() -> None:
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        title = data.get("title", "Unknown title")
        url = data.get("url", "N/A")
        date_crawled = data.get("date_crawled", "N/A")
        content = data.get("content_markdown") or data.get("content") or ""

        header = [
            f"# {title}",
            "",
            f"**Source:** {url}",
            f"**Crawled:** {date_crawled}",
            "",
            "---",
            "",
        ]
        output_path = output_dir / f"{filepath.stem}.md"
        _write_markdown(filepath, output_path, "\n".join(header) + content)


def convert_all() -> None:
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
