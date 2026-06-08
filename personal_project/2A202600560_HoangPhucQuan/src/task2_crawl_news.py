"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    # Ca sĩ Miu Lê bị bắt quả tang dùng ma túy ở bãi biển (VnExpress)
    "https://vnexpress.net/ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien-5072657.html",
    # Ca sĩ Miu Lê bị bắt với cáo buộc tổ chức sử dụng ma túy (VnExpress)
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    # Bắt ca sĩ Long Nhật và Sơn Ngọc Minh vì liên quan ma túy (Tuổi Trẻ)
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    # Ca sĩ Long Nhật khai sử dụng ma túy đá cùng quản lý (Tuổi Trẻ)
    "https://tuoitre.vn/ca-si-long-nhat-khai-su-dung-ma-tuy-da-cung-quan-ly-20260520132251413.htm",
    # Ca sĩ Long Nhật và Sơn Ngọc Minh bị bắt liên quan ma túy (Thanh Niên)
    "https://thanhnien.vn/ca-si-long-nhat-va-son-ngoc-minh-bi-bat-lien-quan-ma-tuy-185260520134218541.htm",
    # Ca sĩ Sơn Ngọc Minh vừa bị bắt vì liên quan đến ma túy là ai? (Thanh Niên)
    "https://thanhnien.vn/ca-si-son-ngoc-minh-vua-bi-bat-vi-lien-quan-den-ma-tuy-la-ai-18526052012481811.htm",
    # Ca sĩ Châu Việt Cường hầu tòa vì nhét tỏi hại chết cô gái 20 tuổi (VnExpress)
    "https://vnexpress.net/ca-si-chau-viet-cuong-hau-toa-vi-nhet-toi-hai-chet-co-gai-20-tuoi-3890738.html",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        # result.markdown là MarkdownGenerationResult, dùng .raw_markdown để lấy text
        markdown_content = ""
        if result.markdown:
            md = result.markdown
            if hasattr(md, "raw_markdown"):
                markdown_content = md.raw_markdown or ""
            elif isinstance(md, str):
                markdown_content = md

        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title") or "Unknown"

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": markdown_content,
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as e:
            print(f"  [ERR] Error: {e}")
            article = {
                "url": url,
                "title": "Unknown",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": "",
                "error": str(e),
            }

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARN] Hay dien ARTICLE_URLS truoc khi chay!")
        print("Goi y: tim bai bao tren VnExpress, Tuoi Tre, Thanh Nien, ...")
    else:
        asyncio.run(crawl_all())
