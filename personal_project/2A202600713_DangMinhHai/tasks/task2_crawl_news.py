"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Chạy:
    python -m src.task2_crawl_news
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


ARTICLE_URLS = [
    "https://tienphong.vn/lien-tiep-nghe-si-dung-chat-cam-post1842599.tpo",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://nld.com.vn/showbiz-viet-nhung-nghe-si-gay-soc-vi-be-boi-ma-tuy-196250725113547841.htm",
    "https://tuoitre.vn/ca-si-chi-dan-nguoi-mau-an-tay-co-tien-truc-phuong-to-chuc-su-dung-ma-tuy-ra-sao-2026040214370414.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
]


def setup_directory() -> None:
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-") or "article"


def _extract_attr(obj, *names, default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value:
            return value
    return default


def _extract_markdown(result) -> str:
    """Trích markdown từ Crawl4AI result theo nhiều kiểu trả về khác nhau."""
    candidates = [
        getattr(result, "markdown", None),
        getattr(result, "cleaned_markdown", None),
        getattr(result, "fit_markdown", None),
        getattr(result, "text", None),
        getattr(result, "content", None),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

        for nested_name in ("raw_markdown", "markdown", "text", "content"):
            nested = getattr(candidate, nested_name, None)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

    return ""


def _extract_title(result, url: str, markdown: str = "") -> str:
    domain = urlparse(url).netloc.lower()

    def is_generic(value: str) -> bool:
        value = value.strip().lower()
        return value in {domain, f"www.{domain}"} or len(value) <= 12

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("title", "og:title", "twitter:title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip() and not is_generic(value):
                return value.strip()

    title = _extract_attr(result, "title", "page_title", default="")
    if isinstance(title, str) and title.strip() and not is_generic(title):
        return title.strip()

    if markdown:
        for line in markdown.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                if not is_generic(line):
                    return line
                break

    return urlparse(url).netloc


def _headline_from_markdown(markdown: str, url: str) -> str:
    """Lấy headline đầu tiên có vẻ là tiêu đề thật từ markdown."""
    domain = urlparse(url).netloc.lower()
    for line in markdown.splitlines():
        candidate = line.strip().lstrip("#").strip()
        if len(candidate) < 20:
            continue
        if candidate.lower() in {domain, f"www.{domain}"}:
            continue
        if candidate.lower().startswith(("podcast", "youtube", "cần biết")):
            continue
        return candidate
    return ""


def _fallback_plain_text(url: str) -> str:
    """Fallback khi crawler không trả về markdown đủ tốt."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:20000]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str,
            "content": str,
            "metadata": dict
        }
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        ignore_https_errors=True,
        enable_stealth=True,
        verbose=False,
    )
    run_config = CrawlerRunConfig(
        wait_until="domcontentloaded",
        page_timeout=90000,
        remove_consent_popups=True,
        remove_overlay_elements=True,
        verbose=False,
        cache_mode=CacheMode.BYPASS,
    )

    result = None
    title = ""
    metadata = {}

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
        markdown = _extract_markdown(result)
    except Exception as exc:
        markdown = ""
        metadata = {"crawler_error": str(exc)}

    if not markdown or len(markdown) < 500:
        markdown = _fallback_plain_text(url)

    if result is not None:
        title = _extract_title(result, url, markdown)

    fallback_title = _headline_from_markdown(markdown, url)
    if fallback_title and (
        not title
        or title.lower() in {urlparse(url).netloc.lower(), f"www.{urlparse(url).netloc.lower()}"}
        or len(title) <= 12
    ):
        title = fallback_title

    if not title:
        title = urlparse(url).netloc

    crawled_at = datetime.now(timezone.utc).isoformat()
    article_metadata = {
        "url": url,
        "title": title,
        "date_crawled": crawled_at,
        "content_markdown": markdown,
        "content": markdown,
        "metadata": {
            "source_url": url,
            "source_domain": urlparse(url).netloc,
            "crawler": "crawl4ai",
            "crawler_metadata": metadata,
        },
    }
    return article_metadata


async def crawl_all() -> None:
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


def main() -> None:
    if not ARTICLE_URLS:
        print("Hãy điền ARTICLE_URLS trước khi chạy.")
        return

    asyncio.run(crawl_all())


if __name__ == "__main__":
    main()
