"""Hybrid search orchestrator combining dense semantic and sparse BM25 search.

This module provides the main interface for Phase 1 hybrid search implementation.
It automatically classifies queries and routes them to the appropriate search strategy.
"""

from typing import List, Dict, Any, Literal, Tuple
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.search.query_classifier import classify_with_override, SearchMode
from app.search.fusion import reciprocal_rank_fusion
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class HybridSearchOrchestrator:
    """Orchestrates hybrid search combining dense + sparse retrieval."""

    def __init__(
        self,
        qdrant: QdrantStorage,
        provider: ProviderRouter
    ):
        """Initialize hybrid search orchestrator.

        Args:
            qdrant: Qdrant storage client
            provider: Provider router for embeddings
        """
        self.qdrant = qdrant
        self.provider = provider

    async def search(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        limit: int = 10,
        search_mode: Literal["auto", "semantic", "keyword", "hybrid"] = "auto",
        filters: Dict[str, Any] | None = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute hybrid search with automatic or manual strategy selection.

        Args:
            query: User search query
            collection_id: Collection UUID as string
            vector_dimension: Vector dimension for embeddings
            limit: Number of results to return
            search_mode: Search mode ("auto", "semantic", "keyword", "hybrid")
            filters: Optional filters

        Returns:
            Tuple of (results, metadata dict with search info)
        """
        # Check if hybrid search is enabled
        if not settings.enable_hybrid_search:
            # Fall back to semantic-only search
            logger.info("hybrid_search_disabled_fallback_to_semantic")
            return await self._search_semantic_only(
                query=query,
                collection_id=collection_id,
                vector_dimension=vector_dimension,
                limit=limit,
                filters=filters
            )

        # Classify query to determine strategy
        classification: SearchMode = classify_with_override(query, search_mode)

        # Detailed logging for flow tracking
        logger.info(
            "🔍 HYBRID_SEARCH_FLOW_STARTED",
            query=query[:150],
            user_requested_mode=search_mode,
            classifier_decision=classification,
            limit=limit,
            collection_id=collection_id
        )

        if search_mode == "auto":
            logger.info(
                f"📊 AUTO_MODE: Classifier selected '{classification}' strategy",
                query_preview=query[:80]
            )
        else:
            logger.info(
                f"👤 MANUAL_OVERRIDE: User forced '{search_mode}' mode, executing '{classification}'",
                override_mode=search_mode
            )

        # Route to appropriate strategy
        if classification == "semantic":
            logger.info("🎯 EXECUTING: Dense semantic search ONLY (no BM25)")
            results, metadata = await self._search_semantic_only(
                query, collection_id, vector_dimension, limit, filters
            )
        elif classification == "keyword_heavy":
            logger.info("🔤 EXECUTING: BM25 keyword search ONLY (no dense vectors)")
            results, metadata = await self._search_keyword_only(
                query, collection_id, limit, filters
            )
        else:  # "hybrid"
            logger.info("🔀 EXECUTING: Hybrid search (Dense + BM25 + RRF fusion)")
            results, metadata = await self._search_hybrid(
                query, collection_id, vector_dimension, limit, filters
            )

        metadata["search_mode_requested"] = search_mode
        metadata["search_mode_executed"] = classification

        logger.info(
            "✅ HYBRID_SEARCH_FLOW_COMPLETED",
            strategy_executed=metadata["strategy"],
            results_returned=len(results),
            top_score=results[0].get("score") or results[0].get("rrf_score") if results else None
        )

        return results, metadata

    async def _search_semantic_only(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        limit: int,
        filters: Dict[str, Any] | None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute semantic search using dense vectors only.

        Args:
            query: User query
            collection_id: Collection ID
            vector_dimension: Vector dimension
            limit: Result limit
            filters: Optional filters

        Returns:
            Tuple of (results, metadata)
        """
        # Generate query embedding
        query_embeddings, provider_name = await self.provider.embed(
            [query],
            required_dimension=vector_dimension
        )
        query_embedding = query_embeddings[0]

        # Search with dense vectors
        results = await self.qdrant.search(
            collection_id=collection_id,
            query_vector=query_embedding,
            limit=limit,
            filters=filters
        )

        metadata = {
            "strategy": "semantic_only",
            "provider": provider_name,
            "results_count": len(results)
        }

        logger.info(
            "🎯 SEMANTIC_SEARCH_COMPLETED",
            results_count=len(results),
            provider=provider_name,
            top_3_scores=[r["score"] for r in results[:3]] if results else []
        )

        return results, metadata

    async def _search_keyword_only(
        self,
        query: str,
        collection_id: str,
        limit: int,
        filters: Dict[str, Any] | None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute keyword search using BM25 sparse vectors only.

        Args:
            query: User query
            collection_id: Collection ID
            limit: Result limit
            filters: Optional filters

        Returns:
            Tuple of (results, metadata)
        """
        # Search with sparse BM25 vectors
        results = await self.qdrant.search_sparse(
            collection_id=collection_id,
            query_text=query,
            limit=limit,
            filters=filters
        )

        metadata = {
            "strategy": "keyword_only",
            "results_count": len(results)
        }

        logger.info(
            "🔤 KEYWORD_SEARCH_COMPLETED",
            results_count=len(results),
            top_3_scores=[r["score"] for r in results[:3]] if results else []
        )

        return results, metadata

    async def _search_hybrid(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        limit: int,
        filters: Dict[str, Any] | None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute hybrid search using both dense and sparse vectors with RRF fusion.

        Args:
            query: User query
            collection_id: Collection ID
            vector_dimension: Vector dimension
            limit: Result limit
            filters: Optional filters

        Returns:
            Tuple of (results, metadata)
        """
        # Run dense and sparse searches in parallel (conceptually - Python async)
        # Step 1: Dense semantic search
        query_embeddings, provider_name = await self.provider.embed(
            [query],
            required_dimension=vector_dimension
        )
        query_embedding = query_embeddings[0]

        dense_results = await self.qdrant.search(
            collection_id=collection_id,
            query_vector=query_embedding,
            limit=limit * 2,  # Retrieve more for fusion
            filters=filters
        )

        # Step 2: Sparse BM25 search
        sparse_results = await self.qdrant.search_sparse(
            collection_id=collection_id,
            query_text=query,
            limit=limit * 2,  # Retrieve more for fusion
            filters=filters
        )

        # Step 3: Reciprocal Rank Fusion
        merged_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=60  # Standard RRF constant
        )

        # Limit to requested number
        final_results = merged_results[:limit]

        metadata = {
            "strategy": "hybrid_rrf",
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "merged_count": len(merged_results),
            "final_count": len(final_results),
            "provider": provider_name
        }

        logger.info(
            "🔀 HYBRID_FUSION_COMPLETED",
            dense_results=len(dense_results),
            sparse_results=len(sparse_results),
            after_rrf_merge=len(merged_results),
            final_returned=len(final_results),
            top_3_rrf_scores=[r["rrf_score"] for r in final_results[:3]] if final_results else [],
            top_result_preview=final_results[0]["content"][:100] if final_results else None
        )

        return final_results, metadata


async def execute_hybrid_search(
    query: str,
    collection_id: str,
    vector_dimension: int,
    qdrant: QdrantStorage,
    provider: ProviderRouter,
    limit: int = 10,
    search_mode: Literal["auto", "semantic", "keyword", "hybrid"] = "auto",
    filters: Dict[str, Any] | None = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience function for executing hybrid search.

    Args:
        query: User search query
        collection_id: Collection UUID
        vector_dimension: Vector dimension
        qdrant: Qdrant storage
        provider: Provider router
        limit: Number of results
        search_mode: Search mode
        filters: Optional filters

    Returns:
        Tuple of (results, metadata)
    """
    orchestrator = HybridSearchOrchestrator(qdrant=qdrant, provider=provider)
    return await orchestrator.search(
        query=query,
        collection_id=collection_id,
        vector_dimension=vector_dimension,
        limit=limit,
        search_mode=search_mode,
        filters=filters
    )
