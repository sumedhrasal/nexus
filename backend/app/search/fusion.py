"""Reciprocal Rank Fusion (RRF) for merging search results from multiple sources.

RRF is a simple but effective algorithm for combining ranked lists from different
retrieval systems (dense semantic search + sparse BM25 search).

Algorithm:
    For each document in any result list:
        RRF_score = sum(1 / (k + rank_i))

    Where:
    - rank_i is the position in result list i (1-indexed)
    - k is a constant (typically 60) to reduce impact of top ranks
    - If document doesn't appear in a list, its contribution is 0

Benefits:
- Rank-based, not score-based (no need to normalize incompatible scores)
- Handles documents appearing in only one list
- Simple consensus voting: documents in both lists get boosted
- Proven effective in hybrid search (used by Elastic, OpenSearch)
"""

from typing import List, Dict, Any
from app.core.logging import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """Merge dense and sparse search results using Reciprocal Rank Fusion.

    Args:
        dense_results: Results from semantic search (ranked by relevance)
        sparse_results: Results from BM25 search (ranked by keyword matching)
        k: RRF constant controlling rank influence (default: 60)

    Returns:
        Merged results sorted by RRF score (highest first)
    """
    rrf_scores: Dict[str, float] = {}
    all_results: Dict[str, Dict[str, Any]] = {}

    # Score dense results (semantic)
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        all_results[chunk_id] = result

    # Score sparse results (BM25)
    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        if chunk_id not in all_results:
            all_results[chunk_id] = result

    # Build merged results with RRF scores
    merged = []
    for chunk_id, rrf_score in rrf_scores.items():
        result = all_results[chunk_id].copy()
        result["rrf_score"] = rrf_score
        # Keep original score as separate field for debugging
        result["original_score"] = all_results[chunk_id]["score"]
        merged.append(result)

    # Sort by RRF score descending
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.debug(
        "rrf_fusion_completed",
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        merged_count=len(merged),
        top_rrf_score=merged[0]["rrf_score"] if merged else 0
    )

    return merged


def weighted_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60
) -> List[Dict[str, Any]]:
    """Merge results with weighted RRF (for query-specific weighting).

    Args:
        dense_results: Results from semantic search
        sparse_results: Results from BM25 search
        dense_weight: Weight for dense/semantic results (0-1)
        sparse_weight: Weight for sparse/BM25 results (0-1)
        k: RRF constant

    Returns:
        Merged results sorted by weighted RRF score
    """
    rrf_scores: Dict[str, float] = {}
    all_results: Dict[str, Dict[str, Any]] = {}

    # Weighted RRF for dense results
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        contribution = dense_weight * (1.0 / (k + rank))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution
        all_results[chunk_id] = result

    # Weighted RRF for sparse results
    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["chunk_id"]
        contribution = sparse_weight * (1.0 / (k + rank))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution
        if chunk_id not in all_results:
            all_results[chunk_id] = result

    # Build merged results
    merged = []
    for chunk_id, rrf_score in rrf_scores.items():
        result = all_results[chunk_id].copy()
        result["rrf_score"] = rrf_score
        result["original_score"] = all_results[chunk_id]["score"]
        merged.append(result)

    # Sort by weighted RRF score
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.debug(
        "weighted_rrf_fusion_completed",
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        merged_count=len(merged)
    )

    return merged
