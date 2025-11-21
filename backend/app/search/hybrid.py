"""Hybrid search combining dense vectors and BM25 sparse scoring."""

from typing import List, Dict, Any, Tuple
from app.search.bm25 import BM25, reciprocal_rank_fusion
import logging

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search using dense vectors and BM25."""

    def __init__(self, alpha: float = 0.5):
        """Initialize hybrid search.

        Args:
            alpha: Weight for dense search (1-alpha for BM25). Default 0.5 for equal weight.
        """
        self.alpha = alpha
        self.bm25 = BM25()

    def combine_results(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_scores: Dict[str, float],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine dense and BM25 results using RRF.

        Args:
            dense_results: Results from dense vector search
            bm25_scores: BM25 scores by chunk_id
            limit: Maximum results to return

        Returns:
            Combined and ranked results
        """
        # FIXED: Use chunk_id instead of entity_id for parent-child chunking
        # Multiple child chunks share the same entity_id (parent document)
        # but have unique chunk_ids
        dense_ranking = [(r["chunk_id"], r["score"]) for r in dense_results]
        bm25_ranking = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply RRF
        fused_scores = reciprocal_rank_fusion([dense_ranking, bm25_ranking])

        # Map back to full results
        result_map = {r["chunk_id"]: r for r in dense_results}
        combined = []

        for chunk_id, rrf_score in fused_scores[:limit]:
            if chunk_id in result_map:
                result = result_map[chunk_id].copy()
                result["score"] = rrf_score  # Use RRF score
                combined.append(result)

        return combined

    def score_with_bm25(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Score documents using BM25.

        Args:
            query: Search query
            documents: List of documents with 'content' field

        Returns:
            Dict mapping chunk_id to BM25 score
        """
        if not documents:
            return {}

        # Extract corpus
        corpus = [doc["content"] for doc in documents]
        # FIXED: Use chunk_id instead of entity_id for parent-child chunking
        chunk_ids = [doc["chunk_id"] for doc in documents]

        # Fit BM25 on corpus
        self.bm25.fit(corpus)

        # Get scores
        scores = self.bm25.get_scores(query, corpus)

        # Map to chunk IDs
        return {chunk_ids[i]: scores[i] for i in range(len(chunk_ids))}


async def hybrid_search(
    query: str,
    dense_results: List[Dict[str, Any]],
    limit: int,
    alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """Perform hybrid search.

    Args:
        query: Search query
        dense_results: Results from dense vector search
        limit: Maximum results to return
        alpha: Weight for dense search (default: 0.5)

    Returns:
        Hybrid search results
    """
    if not dense_results:
        return []

    try:
        # Initialize hybrid search
        hybrid = HybridSearch(alpha=alpha)

        # Calculate BM25 scores
        bm25_scores = hybrid.score_with_bm25(query, dense_results)

        logger.info(f"BM25 scored {len(bm25_scores)} documents")

        # Combine results
        combined = hybrid.combine_results(dense_results, bm25_scores, limit)

        logger.info(f"Hybrid search returned {len(combined)} results")

        return combined

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}, falling back to dense only")
        return dense_results[:limit]
