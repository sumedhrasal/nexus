"""Execution engine router for adaptive RAG strategies.

This module routes query execution to the appropriate retrieval strategy
based on the execution plan.
"""

from typing import List, Dict, Any, Optional, Tuple
from app.search.plan_schema import ExecutionPlan, ExecutionStrategy
from app.search.query_decomposition import get_query_decomposer
from app.search.decomposed_retrieval import retrieve_with_decomposition
from app.search.iterative_retrieval import retrieve_iteratively
from app.search.plan_metrics import log_plan_execution_start, log_plan_execution_complete
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.search.reranker import get_reranker
from app.search.ranking import maximal_marginal_relevance
from app.config import settings
from app.core.logging import get_logger
import time

logger = get_logger(__name__)


class ExecutionEngine:
    """Routes and executes queries based on execution plan."""

    def __init__(
        self,
        qdrant: QdrantStorage,
        provider: ProviderRouter
    ):
        """Initialize execution engine.

        Args:
            qdrant: Qdrant storage client
            provider: Provider router for embeddings and LLM
        """
        self.qdrant = qdrant
        self.provider = provider

    async def execute(
        self,
        plan: ExecutionPlan,
        query: str,
        collection_id: str,
        vector_dimension: int,
        final_limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute query based on execution plan.

        Args:
            plan: Execution plan
            query: User query
            collection_id: Collection to search
            vector_dimension: Expected vector dimension
            final_limit: Final number of results to return
            filters: Optional filters

        Returns:
            Tuple of (results, execution_metadata)
        """
        start_time = time.time()

        log_plan_execution_start(plan, query)

        try:
            # Route to appropriate strategy
            if plan.strategy == ExecutionStrategy.DIRECT:
                results, metadata = await self._execute_direct(
                    plan=plan,
                    query=query,
                    collection_id=collection_id,
                    vector_dimension=vector_dimension,
                    final_limit=final_limit,
                    filters=filters
                )

            elif plan.strategy == ExecutionStrategy.DECOMPOSE:
                results, metadata = await self._execute_decompose(
                    plan=plan,
                    query=query,
                    collection_id=collection_id,
                    vector_dimension=vector_dimension,
                    final_limit=final_limit,
                    filters=filters
                )

            elif plan.strategy == ExecutionStrategy.ITERATIVE:
                results, metadata = await self._execute_iterative(
                    plan=plan,
                    query=query,
                    collection_id=collection_id,
                    vector_dimension=vector_dimension,
                    final_limit=final_limit,
                    filters=filters
                )

            else:
                raise ValueError(f"Unknown execution strategy: {plan.strategy}")

            # Apply final processing (reranking, MMR, etc.)
            results = await self._post_process_results(
                results=results,
                query=query,
                final_limit=final_limit
            )

            execution_time_ms = (time.time() - start_time) * 1000

            log_plan_execution_complete(
                plan=plan,
                query=query,
                execution_time_ms=execution_time_ms,
                success=True
            )

            metadata["execution_time_ms"] = execution_time_ms
            metadata["final_result_count"] = len(results)

            logger.info(
                "execution_completed",
                strategy=plan.strategy.value,
                execution_time_ms=execution_time_ms,
                result_count=len(results)
            )

            return results, metadata

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            log_plan_execution_complete(
                plan=plan,
                query=query,
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )

            logger.error(
                "execution_failed",
                strategy=plan.strategy.value,
                error=str(e),
                exc_info=True
            )

            raise

    async def _execute_direct(
        self,
        plan: ExecutionPlan,
        query: str,
        collection_id: str,
        vector_dimension: int,
        final_limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute DIRECT strategy (single retrieval).

        Args:
            plan: Execution plan
            query: User query
            collection_id: Collection to search
            vector_dimension: Vector dimension
            final_limit: Final limit
            filters: Optional filters

        Returns:
            Tuple of (results, metadata)
        """
        logger.debug(
            "direct_strategy_started",
            initial_retrieval_limit=plan.initial_retrieval_limit
        )

        # Single retrieval with plan's limit
        query_embeddings, provider_name = await self.provider.embed(
            [query],
            required_dimension=vector_dimension
        )
        query_embedding = query_embeddings[0]

        results = await self.qdrant.search(
            collection_id=collection_id,
            query_vector=query_embedding,
            limit=plan.initial_retrieval_limit,
            filters=filters
        )

        metadata = {
            "strategy": "direct",
            "chunks_retrieved": len(results),
            "provider_used": provider_name
        }

        logger.debug(
            "direct_strategy_completed",
            chunks_retrieved=len(results)
        )

        return results, metadata

    async def _execute_decompose(
        self,
        plan: ExecutionPlan,
        query: str,
        collection_id: str,
        vector_dimension: int,
        final_limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute DECOMPOSE strategy (query decomposition + multi-query retrieval).

        Args:
            plan: Execution plan
            query: User query
            collection_id: Collection to search
            vector_dimension: Vector dimension
            final_limit: Final limit
            filters: Optional filters

        Returns:
            Tuple of (results, metadata)
        """
        logger.debug(
            "decompose_strategy_started",
            num_sub_queries=plan.num_sub_queries
        )

        # Step 1: Decompose query
        decomposer = get_query_decomposer(provider=self.provider)
        sub_queries = await decomposer.decompose(
            query=query,
            num_sub_queries=plan.num_sub_queries,
            context=plan.reasoning
        )

        # Step 2: Retrieve for each sub-query and merge
        results = await retrieve_with_decomposition(
            sub_queries=sub_queries,
            collection_id=collection_id,
            vector_dimension=vector_dimension,
            qdrant=self.qdrant,
            provider=self.provider,
            limit_per_query=plan.initial_retrieval_limit,
            filters=filters
        )

        metadata = {
            "strategy": "decompose_and_synthesize",
            "sub_queries": sub_queries,
            "num_sub_queries": len(sub_queries),
            "chunks_retrieved": len(results)
        }

        logger.debug(
            "decompose_strategy_completed",
            num_sub_queries=len(sub_queries),
            chunks_retrieved=len(results)
        )

        return results, metadata

    async def _execute_iterative(
        self,
        plan: ExecutionPlan,
        query: str,
        collection_id: str,
        vector_dimension: int,
        final_limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute ITERATIVE strategy (iterative retrieval with self-assessment).

        This strategy may also use decomposition if plan requests it.

        Args:
            plan: Execution plan
            query: User query
            collection_id: Collection to search
            vector_dimension: Vector dimension
            final_limit: Final limit
            filters: Optional filters

        Returns:
            Tuple of (results, metadata)
        """
        logger.debug(
            "iterative_strategy_started",
            max_iterations=plan.max_iterations,
            use_decomposition=plan.use_decomposition
        )

        # If also using decomposition, decompose first
        queries_to_process = [query]
        sub_queries_used = None

        if plan.use_decomposition:
            decomposer = get_query_decomposer(provider=self.provider)
            sub_queries_used = await decomposer.decompose(
                query=query,
                num_sub_queries=plan.num_sub_queries,
                context=plan.reasoning
            )
            queries_to_process = sub_queries_used

        # Perform iterative retrieval for each query
        all_results = []
        iterations_performed = []

        for q in queries_to_process:
            chunks, iterations = await retrieve_iteratively(
                query=q,
                collection_id=collection_id,
                vector_dimension=vector_dimension,
                qdrant=self.qdrant,
                provider=self.provider,
                max_iterations=plan.max_iterations,
                initial_limit=plan.initial_retrieval_limit,
                additional_limit=5,
                filters=filters
            )
            all_results.extend(chunks)
            iterations_performed.append(iterations)

        # Deduplicate by entity_id
        seen_ids = set()
        deduplicated = []
        for chunk in all_results:
            if chunk["entity_id"] not in seen_ids:
                deduplicated.append(chunk)
                seen_ids.add(chunk["entity_id"])

        metadata = {
            "strategy": "iterative_deep_dive",
            "iterations_performed": max(iterations_performed) if iterations_performed else 0,
            "sub_queries_used": sub_queries_used,
            "chunks_retrieved": len(deduplicated),
            "use_decomposition": plan.use_decomposition
        }

        logger.debug(
            "iterative_strategy_completed",
            iterations=max(iterations_performed) if iterations_performed else 0,
            chunks_retrieved=len(deduplicated)
        )

        return deduplicated, metadata

    async def _post_process_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        final_limit: int
    ) -> List[Dict[str, Any]]:
        """Post-process results with reranking and MMR.

        Args:
            results: Retrieved results
            query: Original query
            final_limit: Final limit

        Returns:
            Post-processed results
        """
        if not results:
            return results

        # Step 1: Reranking (if enabled)
        if settings.enable_reranking:
            try:
                reranker = get_reranker()
                results = await reranker.rerank(
                    query=query,
                    results=results,
                    top_k=min(settings.reranker_top_k, len(results))
                )
                logger.debug(
                    "reranking_applied",
                    num_results=len(results)
                )
            except Exception as e:
                logger.warning(
                    "reranking_failed",
                    error=str(e),
                    using_original_ranking=True
                )

        # Step 2: MMR for diversity
        if len(results) > final_limit:
            results = maximal_marginal_relevance(
                results,
                limit=final_limit,
                lambda_param=0.7  # 70% relevance, 30% diversity
            )
            logger.debug(
                "mmr_applied",
                final_count=len(results)
            )
        else:
            results = results[:final_limit]

        return results


async def execute_with_plan(
    plan: ExecutionPlan,
    query: str,
    collection_id: str,
    vector_dimension: int,
    qdrant: QdrantStorage,
    provider: ProviderRouter,
    final_limit: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience function for executing with a plan.

    Args:
        plan: Execution plan
        query: User query
        collection_id: Collection to search
        vector_dimension: Vector dimension
        qdrant: Qdrant storage
        provider: Provider router
        final_limit: Final number of results
        filters: Optional filters

    Returns:
        Tuple of (results, execution_metadata)
    """
    engine = ExecutionEngine(qdrant=qdrant, provider=provider)
    return await engine.execute(
        plan=plan,
        query=query,
        collection_id=collection_id,
        vector_dimension=vector_dimension,
        final_limit=final_limit,
        filters=filters
    )
