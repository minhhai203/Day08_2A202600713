"""
Task 7 — Reranking Module.
"""

from __future__ import annotations

import json
import os
import math
import re
from copy import deepcopy

import numpy as np

from .task4_chunking_indexing import _embed_texts


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE))


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng OpenAI nếu có key, fallback sang scorer local hybrid.
    """
    if not candidates:
        return []

    openai_ranked = _rerank_with_openai(query, candidates, top_k=top_k)
    if openai_ranked:
        return openai_ranked

    query_embedding = _embed_texts([query])[0].tolist()
    candidate_embeddings = _embed_texts([c["content"] for c in candidates]).tolist()
    query_tokens = _tokenize(query)

    rescored = []
    for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
        content = candidate.get("content", "")
        content_tokens = _tokenize(content)
        overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        semantic = _cosine(query_embedding, candidate_embedding)
        exact_boost = 1.0 if query.lower() in content.lower() else 0.0
        score = 0.65 * semantic + 0.30 * overlap + 0.05 * exact_boost
        rescored.append({**deepcopy(candidate), "score": float(score), "embedding": candidate_embedding})

    rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored[:top_k]


def _rerank_with_openai(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        payload = []
        for idx, candidate in enumerate(candidates, 1):
            payload.append(
                {
                    "id": idx,
                    "source": candidate.get("metadata", {}).get("citation_label")
                    or candidate.get("metadata", {}).get("source")
                    or f"doc_{idx}",
                    "content": candidate.get("content", "")[:1400],
                    "initial_score": float(candidate.get("score", 0.0)),
                }
            )

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict reranker for Vietnamese RAG. "
                        "Score each candidate by relevance to the query. "
                        "Return JSON only as {\"results\":[{\"id\":1,\"score\":87,\"reason\":\"...\"}, ...]}. "
                        "Use scores from 0 to 100. Keep the order sorted by descending score."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"query": query, "candidates": payload}, ensure_ascii=False),
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
        id_to_candidate = {idx + 1: deepcopy(candidate) for idx, candidate in enumerate(candidates)}
        scored = []
        for item in data.get("results", [])[:top_k]:
            candidate = id_to_candidate.get(int(item.get("id", 0)))
            if not candidate:
                continue
            candidate["score"] = float(item.get("score", 0)) / 100.0
            scored.append(candidate)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]
    except Exception:
        return []


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    """
    if not candidates:
        return []

    candidate_embeddings = []
    candidate_list = []
    for candidate in candidates:
        emb = candidate.get("embedding")
        candidate_list.append(candidate)
        candidate_embeddings.append(emb if emb is not None else _embed_texts([candidate["content"]])[0].tolist())

    selected: list[int] = []
    remaining = list(range(len(candidate_list)))

    for _ in range(min(top_k, len(candidate_list))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine(query_embedding, candidate_embeddings[idx])
            diversity = 0.0
            for sel_idx in selected:
                diversity = max(diversity, _cosine(candidate_embeddings[idx], candidate_embeddings[sel_idx]))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = deepcopy(candidate_list[idx])
        item["score"] = float(lambda_param * _cosine(query_embedding, candidate_embeddings[idx]))
        item["embedding"] = candidate_embeddings[idx]
        results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = deepcopy(item)

    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = deepcopy(content_map[content])
        item["score"] = float(score)
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        query_embedding = _embed_texts([query])[0].tolist()
        return rerank_mmr(query_embedding, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
