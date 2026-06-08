"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown pymupdf

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import os
import sys
from pathlib import Path

import pytesseract
from markitdown import MarkItDown

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cấu hình Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Users\Lenovo\tessdata"

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _ocr_pdf(filepath: Path, lang: str = "vie+eng") -> str:
    """OCR từng trang PDF dạng ảnh bằng pytesseract + PyMuPDF."""
    import fitz  # pymupdf
    from PIL import Image
    import io

    doc = fitz.open(str(filepath))
    parts = []
    for page_num, page in enumerate(doc, 1):
        # Render page thành ảnh độ phân giải 300 DPI
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
        if text.strip():
            parts.append(f"## Trang {page_num}\n\n{text.strip()}")
        print(f"    OCR page {page_num}/{len(doc)}...", end="\r")
    doc.close()
    print()
    return "\n\n".join(parts)


def _extract_pdf_with_pymupdf(filepath: Path) -> str:
    """Extract text từ PDF text-based bằng PyMuPDF (không cần OCR)."""
    import fitz  # pymupdf

    doc = fitz.open(str(filepath))
    parts = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            parts.append(f"## Trang {page_num}\n\n{text.strip()}")
    doc.close()
    return "\n\n".join(parts)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        content = ""

        try:
            result = md.convert(str(filepath))
            content = result.text_content or ""
        except Exception as e:
            print(f"  [WARN] MarkItDown failed ({e}), trying pymupdf...")

        # Fallback: nếu content quá ngắn → PDF là ảnh, dùng OCR
        if len(content.strip()) < 200 and filepath.suffix.lower() == ".pdf":
            # Kiểm tra xem PDF có text-layer hay không
            import fitz
            doc = fitz.open(str(filepath))
            has_images = any(doc[0].get_images() for _ in [None])
            doc.close()
            if has_images:
                print("  [INFO] PDF dang anh, dung OCR (Tesseract vie+eng)...")
                try:
                    content = _ocr_pdf(filepath)
                except Exception as e2:
                    print(f"  [ERR] OCR that bai: {e2}")
                    content = _extract_pdf_with_pymupdf(filepath)
            else:
                print("  [INFO] Su dung pymupdf extract text...")
                try:
                    content = _extract_pdf_with_pymupdf(filepath)
                except Exception as e2:
                    print(f"  [ERR] pymupdf that bai: {e2}")

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"  [OK] Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

        content = header + data.get("content_markdown", "")
        output_path.write_text(content, encoding="utf-8")
        print(f"  [OK] Saved: {output_path}")


def convert_all():
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
