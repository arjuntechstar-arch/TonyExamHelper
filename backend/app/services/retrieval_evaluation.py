from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def evaluate_ranking(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int,
) -> dict[str, float]:
    """Calculate standard binary-relevance ranking metrics at k."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not relevant_ids:
        raise ValueError("At least one relevant result is required.")

    ranked_ids = retrieved_ids[:k]
    relevant_hits = [chunk_id in relevant_ids for chunk_id in ranked_ids]
    hit_ranks = [rank for rank, is_relevant in enumerate(relevant_hits, start=1) if is_relevant]
    reciprocal_rank = 1 / hit_ranks[0] if hit_ranks else 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant_hits, start=1)
        if is_relevant
    )
    ideal_hits = min(k, len(relevant_ids))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return {
        f"precision@{k}": sum(relevant_hits) / k,
        f"recall@{k}": sum(relevant_hits) / len(relevant_ids),
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / ideal_dcg,
    }


def evaluate_retrieval_benchmark(
    cases: list[dict[str, Any]],
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Run labeled queries through a retriever and report macro ranking metrics."""
    if not cases:
        raise ValueError("At least one benchmark case is required.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    per_query = []
    metric_names: tuple[str, ...] | None = None
    for case in cases:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])
        results = retrieve(query, k)
        metrics = evaluate_ranking(
            [result["chunk"].id for result in results],
            relevant_ids,
            k=k,
        )
        metric_names = tuple(metrics)
        retrieved_ids = [result["chunk"].id for result in results[:k]]
        per_query.append({
            "query": query,
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": sorted(relevant_ids),
            **metrics,
        })

    assert metric_names is not None
    return {
        "query_count": len(per_query),
        "k": k,
        "metrics": {
            name: sum(query[name] for query in per_query) / len(per_query)
            for name in metric_names
        },
        "per_query": per_query,
    }
