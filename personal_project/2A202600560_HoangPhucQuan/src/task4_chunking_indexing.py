"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn:
- Chunking: RecursiveCharacterTextSplitter
  chunk_size=500: vừa đủ 1 điều-khoản pháp luật; không quá dài gây noise
  chunk_overlap=50: giữ ngữ cảnh liên tục giữa các chunk kề nhau
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
  Chạy local, không cần API key, đủ tốt cho semantic matching
- Vector Store: ChromaDB (persistent local)
  Không cần Docker hay cloud, dễ cài, hỗ trợ cosine similarity
"""

from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "drug_law_docs"

_model: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            if text.strip():
                chunks.append({
                    "content": text,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed chunks bằng all-MiniLM-L6-v2.
    Thêm key 'embedding': list[float] vào mỗi chunk.
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Lưu chunks vào ChromaDB (xoá collection cũ để tránh duplicate)."""
    client = get_chroma_client()

    # Reset collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    global _collection
    _collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        _collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def run_pipeline() -> None:
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print(f"[OK] Indexed to ChromaDB -- {get_collection().count()} chunks total")


if __name__ == "__main__":
    run_pipeline()
