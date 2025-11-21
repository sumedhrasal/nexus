"""Advanced ranking and fusion algorithms for search results."""

from typing import List, Dict, Any
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF is a simple but effective method for combining results from multiple
    retrieval systems. It assigns a score to each document based on its rank
    in each list, which is more robust than score-based fusion.

    Formula: RRF_score(d) = sum over all rankings r: 1 / (k + r(d))

    Args:
        result_lists: List of result lists, each containing dicts with 'chunk_id' and other fields
        k: Constant to avoid division by zero and reduce impact of high ranks (default: 60)

    Returns:
        Merged and re-ranked results as a single list

    Example:
        Query 1 results: [A, B, C]
        Query 2 results: [B, D, A]
        RRF scores: A=1/(60+1)+1/(60+3)=0.032, B=1/(60+2)+1/(60+1)=0.032, C=0.016, D=0.016
        Final ranking: A, B, C, D (or B, A depending on tiebreaker)
    """
    if not result_lists:
        return []

    if len(result_lists) == 1:
        return result_lists[0]

    # Calculate RRF scores
    # FIXED: Use chunk_id instead of entity_id for parent-child chunking
    # Multiple child chunks share the same entity_id but have unique chunk_ids
    rrf_scores: Dict[str, float] = {}
    chunk_data: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result["chunk_id"]

            # Add RRF score contribution from this ranking
            score_contribution = 1.0 / (k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_contribution

            # Store chunk data (use first occurrence)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

    # Sort by RRF score
    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build final result list with RRF scores
    results = []
    for chunk_id, rrf_score in ranked_chunks:
        result = chunk_data[chunk_id].copy()
        result["score"] = rrf_score  # Replace original score with RRF score
        result["ranking_method"] = "rrf"
        results.append(result)

    logger.debug(
        "rrf_fusion_completed",
        num_lists=len(result_lists),
        unique_chunks=len(results),
        top_score=results[0]["score"] if results else 0
    )

    return results


def maximal_marginal_relevance(
    results: List[Dict[str, Any]],
    limit: int,
    lambda_param: float = 0.5
) -> List[Dict[str, Any]]:
    """Re-rank results using Maximal Marginal Relevance (MMR) to reduce redundancy.

    MMR balances relevance and diversity by penalizing results that are too
    similar to already selected results.

    Formula: MMR = argmax[Di in R\S] [λ*Sim1(Di,Q) - (1-λ)*max(Sim2(Di,Dj) for Dj in S)]

    Where:
    - Sim1 = relevance similarity (query-document)
    - Sim2 = diversity similarity (document-document)
    - λ = balance parameter (0=max diversity, 1=max relevance)

    Args:
        results: List of search results with 'score' and 'content' fields
        limit: Number of results to return
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)

    Returns:
        Re-ranked results with reduced redundancy
    """
    if not results or limit >= len(results):
        return results[:limit]

    # Initialize with highest scoring result
    selected = [results[0]]
    remaining = results[1:]

    while len(selected) < limit and remaining:
        # For each remaining result, compute MMR score
        mmr_scores = []

        for candidate in remaining:
            # Relevance score (original search score)
            relevance = candidate["score"]

            # Compute similarity to already selected results
            # Using simple content overlap as proxy for semantic similarity
            max_similarity = 0.0
            for selected_doc in selected:
                similarity = _compute_content_similarity(
                    candidate["content"],
                    selected_doc["content"]
                )
                max_similarity = max(max_similarity, similarity)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((candidate, mmr_score))

        # Select candidate with highest MMR score
        best_candidate, best_score = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    logger.debug(
        "mmr_reranking_completed",
        original_count=len(results),
        selected_count=len(selected),
        lambda_param=lambda_param
    )

    return selected


def _compute_content_similarity(text1: str, text2: str) -> float:
    """Compute simple Jaccard similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Simple word-based Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


def boost_by_metadata(
    results: List[Dict[str, Any]],
    boosts: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Boost results based on metadata fields.

    Args:
        results: Search results
        boosts: Dict mapping metadata field values to boost multipliers
                e.g., {"chunk_type": {"abstract": 1.5, "intro": 1.2}}

    Returns:
        Results with adjusted scores
    """
    boosted = []

    for result in results:
        score = result["score"]
        metadata = result.get("metadata", {})

        # Apply boosts
        for field, boost_map in boosts.items():
            if field in metadata:
                value = metadata[field]
                if value in boost_map:
                    score *= boost_map[value]
                    logger.debug(
                        "result_boosted",
                        entity_id=result["entity_id"],
                        field=field,
                        value=value,
                        multiplier=boost_map[value]
                    )

        result_copy = result.copy()
        result_copy["score"] = score
        boosted.append(result_copy)

    # Re-sort by adjusted scores
    boosted.sort(key=lambda x: x["score"], reverse=True)

    return boosted
