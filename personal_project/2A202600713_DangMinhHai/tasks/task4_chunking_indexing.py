"""
Task 4 — Chunking & Indexing vào Vector Store.

Triển khai local-first với OpenAI embeddings và cache JSON trong repo.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CACHE_DIR = PROJECT_DIR / "data" / ".cache"
INDEX_CACHE_PATH = CACHE_DIR / "task4_index.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

VECTOR_STORE = "local_json"


_INDEX_CACHE: list[dict] | None = None
_BM25_CACHE = None
_BM25_CORPUS: list[dict] | None = None
_LOCAL_VECTOR_TOKENIZER = None

LEGAL_TITLE_MAP = {
    "luat-phong-chong-ma-tuy-2021": "Luật Phòng, chống ma tuý 2021",
    "nghi-dinh-105-2021-nd-cp": "Nghị định 105/2021/NĐ-CP",
    "luat-120-2025": "Luật 120/2025",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _normalize_legal_text(text: str) -> str:
    text = _normalize_whitespace(text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if re.match(r"^(Chương|Điều|Mục|Phần)\s+\w+", line, flags=re.IGNORECASE) and not line.startswith("#"):
            lines.append("")
            lines.append(f"## {line}")
            lines.append("")
        else:
            lines.append(line)
    return _normalize_whitespace("\n".join(lines))


def _extract_doc_title(content: str, source: str) -> str:
    stem = Path(source).stem
    if stem in LEGAL_TITLE_MAP:
        return LEGAL_TITLE_MAP[stem]

    first_lines = [line.strip("# ").strip() for line in content.splitlines() if line.strip()]
    for line in first_lines[:8]:
        if len(line) < 4:
            continue
        if line.startswith(("Source:", "Crawled:", "---", "|")):
            continue
        if any(token in line.lower() for token in ("luật", "nghị định", "bộ luật", "thông tư", "quy định", "nghĩa vụ", "quyền")):
            return line
        if len(line.split()) <= 12:
            return line
    return stem.replace("-", " ").replace("_", " ").title()


def _extract_news_metadata(content: str) -> tuple[str | None, str | None]:
    title = None
    year = None
    for line in content.splitlines()[:12]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# ") and not title:
            title = stripped.lstrip("# ").strip()
            continue
        if stripped.lower().startswith("**crawled:**"):
            year_match = re.search(r"(19|20)\d{2}", stripped)
            if year_match:
                year = year_match.group(0)
        if stripped.lower().startswith("**source:**") and not title:
            source_match = stripped.split("**Source:**", 1)[-1].strip()
            if source_match:
                title = source_match.split("/")[-1]
    return title, year


def _slug_to_title(source: str) -> str:
    stem = Path(source).stem
    return stem.replace("-", " ").replace("_", " ").title()


def _source_type_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    return "legal" if "legal" in parts else "news"


def _source_label(path: Path) -> str:
    rel = path.relative_to(STANDARDIZED_DIR)
    return str(rel).replace("\\", "/")


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    if not STANDARDIZED_DIR.exists():
        return []

    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = _source_type_from_path(md_file)
        if doc_type == "legal":
            content = _normalize_legal_text(content)
        source = _source_label(md_file)
        doc_title = _extract_doc_title(content, source)
        news_title, news_year = (None, None)
        if doc_type == "news":
            news_title, news_year = _extract_news_metadata(content)
            if news_title:
                doc_title = news_title
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": source,
                    "type": doc_type,
                    "title": doc_title,
                    "citation_label": f"{doc_title}, {news_year}" if news_year else doc_title,
                    "path": str(md_file),
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks: list[dict] = []
    for doc in documents:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        sections = _split_sections(doc["content"])
        if not sections:
            sections = [("", doc["content"])]

        chunk_index = 0
        for heading, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            base_metadata = {
                **doc["metadata"],
                "section_heading": heading,
                "citation_label": _build_citation_label(doc["metadata"], heading),
            }

            if len(section_text) <= int(CHUNK_SIZE * 1.15):
                search_text = _build_search_text(doc["metadata"], heading, section_text)
                chunks.append(
                    {
                        "content": section_text,
                        "search_text": search_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1
                continue

            splits = splitter.split_text(section_text)
            for split_text in splits:
                split_text = split_text.strip()
                if not split_text:
                    continue
                search_text = _build_search_text(doc["metadata"], heading, split_text)
                chunks.append(
                    {
                        "content": split_text,
                        "search_text": search_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1
    return chunks


def _split_sections(content: str) -> list[tuple[str, str]]:
    lines = _normalize_whitespace(content).splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush():
        nonlocal current_heading, current_lines
        if current_lines:
            sections.append((current_heading, current_lines[:]))
            current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue
        is_heading = stripped.startswith("#") or re.match(
            r"^(Chương|Điều|Mục|Phần)\s+\w+", stripped, flags=re.IGNORECASE
        )
        if is_heading:
            flush()
            current_heading = stripped.lstrip("# ").strip()
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    flush()
    if not sections:
        return []
    joined = []
    for heading, lines_list in sections:
        joined.append((heading, "\n".join(lines_list).strip()))
    return joined


def _build_citation_label(metadata: dict, heading: str) -> str:
    title = metadata.get("title") or metadata.get("source") or "Nguồn"
    heading = (heading or "").strip()
    if heading:
        if heading.lower().startswith(("điều", "chương", "mục", "phần")):
            return f"{title}, {heading}"
        return f"{title} - {heading}"
    return title


def _build_search_text(metadata: dict, heading: str, content: str) -> str:
    title = metadata.get("title") or metadata.get("source") or ""
    parts = [title, heading, content]
    return "\n".join(part for part in parts if part).strip()


def _get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _local_embedding_matrix(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    global _LOCAL_VECTOR_TOKENIZER
    if _LOCAL_VECTOR_TOKENIZER is None:
        _LOCAL_VECTOR_TOKENIZER = HashingVectorizer(
            n_features=EMBEDDING_DIM,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
    matrix = _LOCAL_VECTOR_TOKENIZER.transform(texts)
    return matrix.toarray().astype(float)


def _embed_texts(texts: list[str]) -> np.ndarray:
    client = _get_openai_client()
    if client is not None:
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            vectors = [item.embedding for item in response.data]
            return np.asarray(vectors, dtype=float)
        except Exception:
            pass
    return _local_embedding_matrix(texts)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return []

    texts = [chunk.get("search_text") or chunk["content"] for chunk in chunks]
    embeddings = _embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.astype(float).tolist()
    return chunks


def _source_signature() -> dict:
    signature = {}
    if not STANDARDIZED_DIR.exists():
        return signature

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        stat = md_file.stat()
        signature[str(md_file)] = [stat.st_mtime_ns, stat.st_size]
    return signature


def _load_cached_index() -> list[dict] | None:
    if not INDEX_CACHE_PATH.exists():
        return None

    try:
        payload = json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    if payload.get("signature") != _source_signature():
        return None

    return payload.get("chunks", [])


def _save_cached_index(chunks: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"signature": _source_signature(), "chunks": chunks}
    INDEX_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào cache local để phục vụ semantic search.
    """
    _save_cached_index(chunks)
    return chunks


def build_index(force: bool = False) -> list[dict]:
    """Load, chunk, embed và cache toàn bộ tài liệu."""
    global _INDEX_CACHE, _BM25_CACHE, _BM25_CORPUS

    if not force and _INDEX_CACHE is not None:
        return _INDEX_CACHE

    if not force:
        cached = _load_cached_index()
        if cached is not None:
            _INDEX_CACHE = cached
            _BM25_CACHE = None
            _BM25_CORPUS = None
            return _INDEX_CACHE

    documents = load_documents()
    if not documents:
        _INDEX_CACHE = []
        return _INDEX_CACHE

    chunks = chunk_documents(documents)
    chunks = embed_chunks(chunks)
    index_to_vectorstore(chunks)

    _INDEX_CACHE = chunks
    _BM25_CACHE = None
    _BM25_CORPUS = None
    return chunks


def _get_index() -> list[dict]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = build_index(force=False)
    return _INDEX_CACHE or []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.
    """
    chunks = _get_index()
    if not chunks:
        return []

    query_embedding = _embed_texts([query])[0].tolist()
    query_tokens = set(_tokenize(query))

    scored = []
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            continue
        search_text = chunk.get("search_text") or chunk["content"]
        content_tokens = set(_tokenize(search_text))
        overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        exact_boost = 0.18 if query.lower() in search_text.lower() else 0.0
        article_boost = 0.0
        article_match = re.search(r"điều\s+(\d+)", query, flags=re.IGNORECASE)
        if article_match and re.search(
            rf"điều\s*{re.escape(article_match.group(1))}\b", search_text, flags=re.IGNORECASE
        ):
            article_boost = 0.2
        score = 0.72 * _cosine_similarity(query_embedding, embedding) + 0.18 * overlap + exact_boost + article_boost
        scored.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "embedding": embedding,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc.get("search_text") or doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _ensure_bm25():
    global _BM25_CACHE, _BM25_CORPUS
    corpus = _get_index()
    if _BM25_CACHE is None or _BM25_CORPUS is not corpus:
        _BM25_CORPUS = corpus
        _BM25_CACHE = build_bm25_index(corpus) if corpus else None
    return _BM25_CACHE, _BM25_CORPUS or []


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.
    """
    bm25, corpus = _ensure_bm25()
    if bm25 is None or not corpus:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = []
    for idx in np.argsort(scores)[::-1][:top_k]:
        score = float(scores[idx])
        if score <= 0 and ranked:
            continue
        corpus_item = corpus[idx]
        search_text = corpus_item.get("search_text") or corpus_item["content"]
        if query.lower() in search_text.lower():
            score += 0.25
        article_match = re.search(r"điều\s+(\d+)", query, flags=re.IGNORECASE)
        if article_match and re.search(
            rf"điều\s*{re.escape(article_match.group(1))}\b", search_text, flags=re.IGNORECASE
        ):
            score += 0.4
        ranked.append(
            {
                "content": corpus_item["content"],
                "score": score,
                "metadata": corpus_item.get("metadata", {}),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
