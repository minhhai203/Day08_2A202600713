from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from .embeddings import OPENAI_EMBEDDING_DIMENSIONS, embed_text, embed_texts

load_dotenv()


@dataclass(slots=True)
class IngestResult:
    document_id: str
    document_title: str
    chunk_count: int
    target_mode: str


class PostgresVectorStore:
    def __init__(self) -> None:
        self.dbname = os.getenv("POSTGRES_DB", "chatbot-rag")
        self.user = os.getenv("POSTGRES_USER") or None
        self.password = os.getenv("POSTGRES_PASSWORD") or None
        self.host = os.getenv("POSTGRES_HOST") or None
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))

    def _connect(self, *, register: bool = True) -> psycopg.Connection:
        conn = psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            autocommit=False,
        )
        if register:
            register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        with self._connect(register=False) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                register_vector(conn)
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS rag_documents (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        source TEXT NOT NULL,
                        source_type TEXT NOT NULL DEFAULT 'db',
                        url TEXT NOT NULL DEFAULT '',
                        checksum TEXT NOT NULL DEFAULT '',
                        raw_content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding VECTOR({OPENAI_EMBEDDING_DIMENSIONS}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS rag_chunks_document_idx
                    ON rag_chunks (document_id, chunk_index);
                    """
                )
            conn.commit()

    def ingest_document(
        self,
        title: str,
        source: str,
        content: str,
        chunks: list[str],
        metadata: dict[str, Any] | None = None,
        url: str = "",
    ) -> IngestResult:
        self.ensure_schema()
        checksum = hashlib.sha1(content.encode("utf-8")).hexdigest()
        document_id = f"doc_{hashlib.sha1(source.encode('utf-8')).hexdigest()[:16]}"
        vectors = embed_texts(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding result count does not match chunk count.")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_chunks WHERE document_id = %s", (document_id,))
                cur.execute(
                    """
                    INSERT INTO rag_documents (id, title, source, source_type, url, checksum, raw_content, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source = EXCLUDED.source,
                        source_type = EXCLUDED.source_type,
                        url = EXCLUDED.url,
                        checksum = EXCLUDED.checksum,
                        raw_content = EXCLUDED.raw_content,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW();
                    """,
                    (
                        document_id,
                        title,
                        source,
                        "db",
                        url,
                        checksum,
                        content,
                        psycopg.types.json.Json(metadata or {}),
                    ),
                )
                for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                    chunk_id = f"{document_id}_chunk_{idx}"
                    cur.execute(
                        """
                        INSERT INTO rag_chunks (id, document_id, chunk_index, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata;
                        """,
                        (
                            chunk_id,
                            document_id,
                            idx,
                            chunk,
                            vector,
                            psycopg.types.json.Json({"title": title, "source": source, **(metadata or {})}),
                        ),
                    )
            conn.commit()

        return IngestResult(
            document_id=document_id,
            document_title=title,
            chunk_count=len(chunks),
            target_mode="db",
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.ensure_schema()
        query_vector = embed_text(query)
        if not query_vector:
            return []
        vector_param = Vector(query_vector)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        c.id,
                        c.content,
                        1 - (c.embedding <=> %s) AS score,
                        d.title,
                        d.source,
                        d.url,
                        d.metadata
                    FROM rag_chunks c
                    JOIN rag_documents d ON d.id = c.document_id
                    ORDER BY c.embedding <=> %s
                    LIMIT %s;
                    """,
                    (vector_param, vector_param, top_k),
                )
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            chunk_id, content, score, title, source, url, metadata = row
            payload = dict(metadata or {})
            payload.update(
                {
                    "title": title,
                    "citation_label": title,
                    "source": source,
                    "source_url": url or payload.get("source_url", ""),
                    "chunk_id": chunk_id,
                    "type": "db-upload",
                }
            )
            results.append(
                {
                    "content": content,
                    "score": float(score or 0.0),
                    "metadata": payload,
                    "source": "pgvector",
                }
            )
        return results
