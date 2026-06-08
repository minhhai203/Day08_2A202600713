"""
Task 7 — Reranking Module.

Phương pháp chính: Jina Reranker v2 (cross-encoder, multilingual)
- Model: jina-reranker-v2-base-multilingual
- So sánh query-document trực tiếp (cross-encoder) → chính xác hơn bi-encoder
- Multilingual: tốt cho tiếng Việt và văn bản pháp luật
- API endpoint: https://api.jina.ai/v1/rerank

Fallback khi chưa có JINA_API_KEY: RRF (Reciprocal Rank Fusion)
- Không cần model, chỉ dựa trên ranking position
- RRF(d) = Σ 1 / (k + rank(d)), k=60 (Cormack et al. 2009)
"""

from __future__ import annotations

import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng Jina Reranker v2 API.

    Fallback sang RRF nếu JINA_API_KEY chưa được set.
    """
    if not JINA_API_KEY or JINA_API_KEY.startswith("jina_xxx"):
        print("[WARN] JINA_API_KEY chua set -- dung RRF fallback")
        return rerank_rrf([candidates], top_k=top_k)

    documents = [c["content"] for c in candidates]
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    reranked = response.json()["results"]

    return [
        {**candidates[r["index"]], "score": float(r["relevance_score"])}
        for r in reranked
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    λ=0.7: ưu tiên relevance hơn diversity.
    """

    def cosine_sim(a: list, b: list) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            relevance = (
                cosine_sim(query_embedding, emb)
                if emb is not None
                else candidates[idx].get("score", 0.0)
            )

            max_sim = 0.0
            for sel in selected:
                sel_emb = candidates[sel].get("embedding")
                cur_emb = candidates[idx].get("embedding")
                if sel_emb is not None and cur_emb is not None:
                    max_sim = max(max_sim, cosine_sim(cur_emb, sel_emb))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [{**candidates[i], "score": candidates[i].get("score", 0.0)} for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank(d))
    k=60: smoothing constant từ paper Cormack et al. 2009.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {**content_map[content], "score": score}
        for content, score in sorted_items[:top_k]
    ]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        method: 'cross_encoder' | 'rrf' | 'mmr'
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "mmr":
        raise ValueError("MMR cần query_embedding — gọi rerank_mmr() trực tiếp")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
