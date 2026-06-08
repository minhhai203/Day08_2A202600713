"""
Task 4 — Chunking & Indexing vào Vector Store.

Chunking strategy: RecursiveCharacterTextSplitter
    - chunk_size=500: đủ ngữ cảnh cho câu hỏi pháp lý mà không quá dài gây nhiễu
    - chunk_overlap=50: ~10% để bảo toàn ngữ cảnh tại biên giữa các chunk

Embedding model: Google gemini-embedding-2 (via API)
    - Multilingual, 768-dim, dùng chung GEMINI_API_KEY
    - Không cần tải model về máy

Vector Store: ChromaDB (local persistent, không cần Docker hay cloud)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = str(Path(__file__).parent.parent / "data" / "chroma_db")
CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "gemini-embedding-2"  # Google Embedding API
EMBEDDING_DIM = 768

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "drug_law_docs"


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def _recursive_split(text: str, separators: list, chunk_size: int, chunk_overlap: int) -> list[str]:
    """RecursiveCharacterTextSplitter tự implement — không cần langchain dependency."""
    if len(text) <= chunk_size:
        return [text]

    # Tìm separator phù hợp đầu tiên có trong text
    sep = ""
    remaining_seps = []
    for i, s in enumerate(separators):
        if s == "" or s in text:
            sep = s
            remaining_seps = separators[i + 1:]
            break

    parts = text.split(sep) if sep else list(text)

    chunks = []
    current = ""

    for part in parts:
        connector = sep if current else ""
        candidate = current + connector + part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # Giữ overlap: lấy đuôi chunk cũ làm đầu chunk mới
                tail = current[-chunk_overlap:] if chunk_overlap else ""
                current = (tail + sep + part).lstrip(sep) if tail else part
            else:
                # Part quá dài → recurse với separator tiếp theo
                if remaining_seps:
                    sub = _recursive_split(part, remaining_seps, chunk_size, chunk_overlap)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    # Force split theo size
                    for j in range(0, len(part), chunk_size - chunk_overlap):
                        chunks.append(part[j: j + chunk_size])
                    current = ""

    if current and current.strip():
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents bằng RecursiveCharacterTextSplitter (tự implement)."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    for doc in documents:
        splits = _recursive_split(doc["content"], separators, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng Google gemini-embedding-2 API (google-genai SDK)."""
    import time
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    client = genai.Client(api_key=api_key)

    for i, chunk in enumerate(chunks):
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk["content"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        chunk["embedding"] = result.embeddings[0].values
        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks...")
            time.sleep(1)  # tránh rate limit

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
    print(f"  Indexed {len(chunks)} chunks to ChromaDB at {CHROMA_DIR}")


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

    # Lưu chunks (không có embedding) để Task 6 dùng cho BM25
    chunks_for_bm25 = [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    CHUNKS_FILE.write_text(json.dumps(chunks_for_bm25, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Saved chunks to {CHUNKS_FILE} (for BM25)")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
