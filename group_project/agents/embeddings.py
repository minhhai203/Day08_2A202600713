from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    payload = [text.strip() for text in texts if text and text.strip()]
    if not payload:
        return []

    client = _get_openai_client()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=payload,
    )
    return [item.embedding for item in response.data]


def embed_text(text: str) -> list[float]:
    vectors = embed_texts([text])
    return vectors[0] if vectors else []
