"""
Task 7 — Reranking Module.

Mặc định: Jina Reranker v2 (cross-encoder, multilingual, tốt cho tiếng Việt).
Fallback: RRF nếu JINA_API_KEY chưa được set.

Cũng có sẵn: MMR và RRF để dùng độc lập.
"""

import os
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker v2 (cross-encoder multilingual).
    Fallback sang RRF nếu JINA_API_KEY không được set.
    """
    if not JINA_API_KEY:
        print("  ⚠ JINA_API_KEY not set — fallback to RRF reranking")
        return rerank_rrf([[c] for c in candidates], top_k=top_k)

    import requests

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k
            },
            timeout=10
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        return [
            {**candidates[r["index"]], "score": r["relevance_score"]}
            for r in reranked
        ]
    except Exception as e:
        print(f"  ⚠ Jina reranker error ({e}) — fallback to RRF")
        return rerank_rrf([[c] for c in candidates], top_k=top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR(d) = λ * sim(query, d) - (1-λ) * max(sim(d, selected_docs))
    lambda_param=0.7: ưu tiên relevance hơn diversity (70/30).
    """
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (denom + 1e-10))

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            if not emb:
                continue
            relevance = cosine_sim(query_embedding, emb)

            max_sim = 0.0
            for sel in selected:
                sel_emb = candidates[sel].get("embedding", [])
                if sel_emb:
                    max_sim = max(max_sim, cosine_sim(emb, sel_emb))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    k=60: smoothing constant từ paper Cormack et al. 2009, giảm ưu thế của
    rank 1 và làm mượt sự chênh lệch giữa các vị trí cao.
    """
    rrf_scores: dict = {}
    content_map: dict = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" | "rrf" | "mmr"
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([[c] for c in candidates], top_k=top_k)
    elif method == "mmr":
        raise ValueError("rerank_mmr cần query_embedding — gọi trực tiếp rerank_mmr()")
    else:
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
