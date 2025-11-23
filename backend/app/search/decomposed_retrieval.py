"""Retrieval engine for decomposed queries.

This module handles retrieval for queries that have been decomposed into
sub-queries, performing sequential retrieval and merging results.
"""

from typing import List, Dict, Any, Optional
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.search.ranking import reciprocal_rank_fusion
from app.core.logging import get_logger

logger = get_logger(__name__)


class DecomposedRetrievalEngine:
    """Handles retrieval for decomposed queries."""

    def __init__(
        self,
        qdrant: QdrantStorage,
        provider: ProviderRouter
    ):
        """Initialize decomposed retrieval engine.

        Args:
            qdrant: Qdrant storage client for vector search
            provider: Provider router for embeddings
        """
        self.qdrant = qdrant
        self.provider = provider

    async def retrieve(
        self,
        sub_queries: List[str],
        collection_id: str,
        vector_dimension: int,
        limit_per_query: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve results for multiple sub-queries and merge them.

        Performs sequential retrieval for each sub-query and uses
        Reciprocal Rank Fusion (RRF) to merge results.

        Args:
            sub_queries: List of sub-queries to search
            collection_id: Collection to search in
            vector_dimension: Expected vector dimension
            limit_per_query: Number of results to retrieve per sub-query
            filters: Optional filters to apply

        Returns:
            Merged and ranked list of results

        Raises:
            Exception: If retrieval fails for all sub-queries
        """
        logger.info(
            "decomposed_retrieval_started",
            collection_id=collection_id,
            num_sub_queries=len(sub_queries),
            limit_per_query=limit_per_query
        )

        all_result_lists: List[List[Dict[str, Any]]] = []
        successful_queries = 0
        failed_queries = 0

        # Retrieve results for each sub-query sequentially
        for idx, sub_query in enumerate(sub_queries):
            try:
                logger.debug(
                    "sub_query_retrieval_started",
                    index=idx,
                    sub_query=sub_query[:100]
                )

                # Embed sub-query
                query_embeddings, provider_name = await self.provider.embed(
                    [sub_query],
                    required_dimension=vector_dimension
                )
                query_embedding = query_embeddings[0]

                # Search
                results = await self.qdrant.search(
                    collection_id=collection_id,
                    query_vector=query_embedding,
                    limit=limit_per_query,
                    filters=filters
                )

                all_result_lists.append(results)
                successful_queries += 1

                logger.debug(
                    "sub_query_retrieval_completed",
                    index=idx,
                    results_found=len(results),
                    provider=provider_name
                )

            except Exception as e:
                logger.error(
                    "sub_query_retrieval_failed",
                    index=idx,
                    sub_query=sub_query[:100],
                    error=str(e),
                    exc_info=True
                )
                failed_queries += 1
                # Continue with other sub-queries

        # Check if we got any results
        if successful_queries == 0:
            logger.error(
                "all_sub_queries_failed",
                total_queries=len(sub_queries)
            )
            raise Exception("All sub-query retrievals failed")

        if failed_queries > 0:
            logger.warning(
                "partial_sub_query_failure",
                successful=successful_queries,
                failed=failed_queries,
                total=len(sub_queries)
            )

        # Merge results using Reciprocal Rank Fusion
        merged_results = self._merge_results(all_result_lists)

        logger.info(
            "decomposed_retrieval_completed",
            num_sub_queries=len(sub_queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            total_unique_results=len(merged_results)
        )

        return merged_results

    def _merge_results(
        self,
        result_lists: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Merge multiple result lists using Reciprocal Rank Fusion.

        Args:
            result_lists: List of result lists from different sub-queries

        Returns:
            Merged and ranked list of results
        """
        if len(result_lists) == 0:
            return []

        if len(result_lists) == 1:
            return sorted(result_lists[0], key=lambda x: x["score"], reverse=True)

        # Use RRF for multi-query fusion
        merged = reciprocal_rank_fusion(result_lists, k=60)

        logger.debug(
            "results_merged",
            num_result_lists=len(result_lists),
            total_unique=len(merged)
        )

        return merged


async def retrieve_with_decomposition(
    sub_queries: List[str],
    collection_id: str,
    vector_dimension: int,
    qdrant: QdrantStorage,
    provider: ProviderRouter,
    limit_per_query: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Convenience function for decomposed retrieval.

    Args:
        sub_queries: List of sub-queries to search
        collection_id: Collection to search in
        vector_dimension: Expected vector dimension
        qdrant: Qdrant storage client
        provider: Provider router for embeddings
        limit_per_query: Number of results per sub-query
        filters: Optional filters

    Returns:
        Merged results
    """
    engine = DecomposedRetrievalEngine(qdrant=qdrant, provider=provider)
    return await engine.retrieve(
        sub_queries=sub_queries,
        collection_id=collection_id,
        vector_dimension=vector_dimension,
        limit_per_query=limit_per_query,
        filters=filters
    )
