"""
Day 8 v2 — RAG Pipeline
Automated Test Suite cho bài cá nhân — Hoàng Phúc Quân (2A202600560).

Chạy từ thư mục personal_project/2A202600560_HoangPhucQuan/:
    pytest tests/test_individual.py -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPO_ROOT = PROJECT_DIR.parent.parent
GROUP_PROJECT_DIR = REPO_ROOT / "group_project"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GROUP_PROJECT_DIR))


# ===========================================================================
# Task 1 — Thu thập văn bản pháp luật
# ===========================================================================

class TestTask1_LegalFiles(unittest.TestCase):
    """Task 1: Thu thập văn bản pháp luật vào data/landing/legal/"""

    def test_landing_legal_dir_exists(self):
        legal_dir = DATA_DIR / "landing" / "legal"
        self.assertTrue(legal_dir.exists(), f"Thư mục không tồn tại: {legal_dir}")

    def test_has_legal_files(self):
        legal_dir = DATA_DIR / "landing" / "legal"
        if not legal_dir.exists():
            self.skipTest("data/landing/legal/ chưa tồn tại")
        valid_ext = {".pdf", ".docx", ".doc"}
        files = [f for f in legal_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_ext]
        self.assertGreaterEqual(
            len(files), 1,
            f"Cần ít nhất 1 file pháp luật, hiện có {len(files)}"
        )

    def test_legal_files_not_empty(self):
        legal_dir = DATA_DIR / "landing" / "legal"
        if not legal_dir.exists():
            self.skipTest("data/landing/legal/ chưa tồn tại")
        valid_ext = {".pdf", ".docx", ".doc"}
        files = [f for f in legal_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_ext]
        for f in files:
            self.assertGreater(
                f.stat().st_size, 1024,
                f"File {f.name} quá nhỏ ({f.stat().st_size} bytes)"
            )


# ===========================================================================
# Task 2 — Crawl bài báo
# ===========================================================================

class TestTask2_NewsFiles(unittest.TestCase):
    """Task 2: Crawl ≥5 bài báo vào data/landing/news/"""

    def test_landing_news_dir_exists(self):
        news_dir = DATA_DIR / "landing" / "news"
        self.assertTrue(news_dir.exists(), f"Thư mục không tồn tại: {news_dir}")

    def test_minimum_5_news_files(self):
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")
        valid_ext = {".json", ".html", ".md", ".txt"}
        files = [f for f in news_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_ext]
        self.assertGreaterEqual(
            len(files), 5,
            f"Cần ≥5 bài báo, hiện có {len(files)}"
        )

    def test_news_files_have_content(self):
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")
        files = [f for f in news_dir.iterdir()
                 if f.is_file() and not f.name.startswith(".")]
        for f in files[:5]:
            self.assertGreater(
                f.stat().st_size, 500,
                f"File {f.name} quá nhỏ, crawl có thể bị lỗi"
            )

    def test_json_files_have_url_field(self):
        import json
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")
        json_files = list(news_dir.glob("*.json"))
        if not json_files:
            self.skipTest("Không có file JSON")
        for f in json_files[:3]:
            data = json.loads(f.read_text(encoding="utf-8"))
            self.assertIn("url", data, f"{f.name} thiếu trường 'url'")


# ===========================================================================
# Task 3 — Convert sang Markdown
# ===========================================================================

class TestTask3_Standardized(unittest.TestCase):
    """Task 3: Convert toàn bộ files sang markdown trong data/standardized/"""

    def test_standardized_dir_exists(self):
        self.assertTrue(
            (DATA_DIR / "standardized").exists(),
            "Thư mục data/standardized/ chưa tồn tại"
        )

    def test_has_markdown_files(self):
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")
        md_files = list(std_dir.rglob("*.md"))
        self.assertGreater(len(md_files), 0, "Không tìm thấy file .md nào")

    def test_converted_files_have_content(self):
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")
        md_files = list(std_dir.rglob("*.md"))
        if not md_files:
            self.skipTest("Chưa có file markdown")
        for f in md_files[:5]:
            content = f.read_text(encoding="utf-8")
            self.assertGreater(
                len(content), 200,
                f"{f.name} quá ngắn ({len(content)} chars)"
            )

    def test_news_standardized_exists(self):
        news_std = DATA_DIR / "standardized" / "news"
        self.assertTrue(
            news_std.exists() and list(news_std.glob("*.md")),
            "data/standardized/news/ không có file .md"
        )

    def test_markdown_has_source_metadata(self):
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")
        md_files = list(std_dir.rglob("*.md"))
        if not md_files:
            self.skipTest("Chưa có file markdown")
        files_with_source = [
            f for f in md_files
            if "**Source:**" in f.read_text(encoding="utf-8")
        ]
        self.assertGreater(
            len(files_with_source), 0,
            "Không có file nào có metadata '**Source:**'"
        )


# ===========================================================================
# Task 6 — Discovery Tools (group_project/agents/discovery_tools.py)
# ===========================================================================

class TestTask6_DiscoveryTools(unittest.TestCase):
    """Task 6 (G6): discovery_tools.py — list/read nguồn."""

    def setUp(self):
        try:
            from agents.discovery_tools import (
                list_available_sources,
                list_sources_with_metadata,
                read_source_text,
            )
            self.list_sources = list_available_sources
            self.list_with_meta = list_sources_with_metadata
            self.read_source = read_source_text
        except ImportError as e:
            self.skipTest(f"Không import được discovery_tools: {e}")

    def test_list_available_sources_returns_list(self):
        result = self.list_sources()
        self.assertIsInstance(result, list)

    def test_list_available_sources_not_empty(self):
        result = self.list_sources()
        self.assertGreater(len(result), 0, "Không tìm thấy nguồn nào")

    def test_list_available_sources_no_hidden_files(self):
        result = self.list_sources()
        for name in result:
            self.assertFalse(
                name.startswith("."),
                f"File ẩn không được list: {name}"
            )

    def test_list_available_sources_is_sorted(self):
        result = self.list_sources()
        self.assertEqual(result, sorted(result), "Danh sách phải được sort")

    def test_read_source_text_existing_file(self):
        sources = self.list_sources()
        if not sources:
            self.skipTest("Không có file nguồn")
        content = self.read_source(sources[0])
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0, "Nội dung file không được rỗng")

    def test_read_source_text_not_found(self):
        result = self.read_source("file_khong_ton_tai_xyz.md")
        self.assertIn("Khong tim thay", result)


# ===========================================================================
# Task 7 — Source Panel QA (list_sources_with_metadata)
# ===========================================================================

class TestTask7_SourcePanel(unittest.TestCase):
    """Task 7 (G7): Source panel — title/source/url/snippet/type rõ ràng."""

    def setUp(self):
        try:
            from agents.discovery_tools import list_sources_with_metadata
            self.list_with_meta = list_sources_with_metadata
        except ImportError as e:
            self.skipTest(f"Không import được discovery_tools: {e}")

    def test_returns_list(self):
        result = self.list_with_meta()
        self.assertIsInstance(result, list)

    def test_not_empty(self):
        result = self.list_with_meta()
        self.assertGreater(len(result), 0, "Không có nguồn nào")

    def test_each_item_has_required_keys(self):
        result = self.list_with_meta()
        if not result:
            self.skipTest("Không có dữ liệu")
        required = {"title", "source", "url", "snippet", "type"}
        for item in result:
            missing = required - item.keys()
            self.assertEqual(
                missing, set(),
                f"Item thiếu keys: {missing}"
            )

    def test_title_not_empty(self):
        result = self.list_with_meta()
        if not result:
            self.skipTest("Không có dữ liệu")
        for item in result:
            self.assertGreater(
                len(item.get("title", "")), 0,
                f"Title rỗng cho source: {item.get('source', '?')}"
            )

    def test_snippet_max_length(self):
        result = self.list_with_meta()
        if not result:
            self.skipTest("Không có dữ liệu")
        for item in result:
            snippet = item.get("snippet", "")
            self.assertLessEqual(
                len(snippet), 300,
                f"Snippet vượt 300 ký tự: {len(snippet)}"
            )

    def test_type_valid_values(self):
        result = self.list_with_meta()
        if not result:
            self.skipTest("Không có dữ liệu")
        valid_types = {"legal", "news", "document"}
        for item in result:
            self.assertIn(
                item.get("type"), valid_types,
                f"Type không hợp lệ: {item.get('type')}"
            )


# ===========================================================================
# Corpus — group_project/agents/corpus.py
# ===========================================================================

class TestCorpus(unittest.TestCase):
    """Kiểm tra corpus.py: load_documents, score_document, search_documents."""

    def setUp(self):
        try:
            from agents.corpus import load_documents, score_document, search_documents
            self.load = load_documents
            self.score = score_document
            self.search = search_documents
        except ImportError as e:
            self.skipTest(f"Không import được corpus: {e}")

    def test_load_documents_returns_list(self):
        docs = self.load()
        self.assertIsInstance(docs, list)

    def test_load_documents_not_empty(self):
        docs = self.load()
        self.assertGreater(len(docs), 0, "Corpus rỗng")

    def test_each_doc_has_content_and_metadata(self):
        docs = self.load()
        if not docs:
            self.skipTest("Corpus rỗng")
        for doc in docs:
            self.assertIn("content", doc)
            self.assertIn("metadata", doc)

    def test_metadata_has_required_fields(self):
        docs = self.load()
        if not docs:
            self.skipTest("Corpus rỗng")
        for doc in docs:
            meta = doc["metadata"]
            for field in ("title", "source", "type"):
                self.assertIn(field, meta, f"metadata thiếu field '{field}'")

    def test_score_empty_query_returns_zero(self):
        docs = self.load()
        if not docs:
            self.skipTest("Corpus rỗng")
        score = self.score("", docs[0])
        self.assertEqual(score, 0.0)

    def test_score_matching_query_positive(self):
        docs = self.load()
        if not docs:
            self.skipTest("Corpus rỗng")
        score = self.score("ma túy", docs[0])
        self.assertGreaterEqual(score, 0.0)

    def test_score_in_range_0_to_1(self):
        docs = self.load()
        if not docs:
            self.skipTest("Corpus rỗng")
        for doc in docs[:5]:
            s = self.score("ma túy phòng chống", doc)
            self.assertGreaterEqual(s, 0.0, "Score phải >= 0")
            self.assertLessEqual(s, 1.0, "Score phải <= 1")

    def test_search_returns_list(self):
        result = self.search("ma túy", top_k=3)
        self.assertIsInstance(result, list)

    def test_search_respects_top_k(self):
        result = self.search("ma túy", top_k=2)
        self.assertLessEqual(len(result), 2)

    def test_search_sorted_descending(self):
        result = self.search("ma túy phòng chống", top_k=5)
        if len(result) < 2:
            self.skipTest("Không đủ kết quả để test sort")
        scores = [r["score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_empty_query_does_not_crash(self):
        result = self.search("", top_k=3)
        self.assertIsInstance(result, list)

    def test_search_results_have_required_keys(self):
        result = self.search("luật phòng chống ma túy", top_k=3)
        if not result:
            self.skipTest("Không có kết quả")
        for r in result:
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("metadata", r)


# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
