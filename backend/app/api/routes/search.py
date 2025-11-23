"""Search endpoints."""

import time
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.database import Collection, SearchAnalytics
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.api.dependencies import get_db, get_qdrant_client, get_embedding_router
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.search.cache import get_cache
from app.search.hybrid import hybrid_search
from app.search.query_expansion import get_expansion_service
from app.search.query_reformulation import get_query_reformulator
from app.search.query_classifier import get_adaptive_weights, classify_query
from app.search.reranker import get_reranker
from app.search.context_expansion import expand_context_windows
from app.search.synthesis import get_synthesis_service
from app.search.ranking import reciprocal_rank_fusion, maximal_marginal_relevance
from app.config import settings
from app.core.logging import get_logger
from app.core.metrics import (
    record_search_metrics,
    record_synthesis_metrics,
    query_expansion_requests_total,
    query_expansion_duration_seconds,
)

router = APIRouter(prefix="/collections/{collection_id}/search", tags=["search"])
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)


@router.post("", response_model=SearchResponse)
@limiter.limit("30/minute")  # 30 searches per minute
async def search_collection(
    request: Request,
    collection_id: uuid.UUID,
    search_request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client)
):
    """Search a collection with optional query expansion and response synthesis."""
    start_time = time.time()

    logger.info(
        "search_started",
        collection_id=str(collection_id),
        query=search_request.query,
        expand_query=search_request.expand_query,
        synthesize=search_request.synthesize,
    )

    # Verify collection exists
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        logger.warning("collection_not_found", collection_id=str(collection_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Get provider router filtered to collection's embedding provider
    from app.api.dependencies import get_collection_router
    router = await get_collection_router(
        collection_id=collection_id,
        provider_override=search_request.provider,
        db=db
    )

    # IMPROVEMENT: Query reformulation (convert natural language to search-friendly)
    reformulated_query = search_request.query
    reformulator = get_query_reformulator()
    try:
        reformulated_query = await reformulator.reformulate(search_request.query, use_llm=True)
        logger.info(
            "query_reformulated",
            original=search_request.query,
            reformulated=reformulated_query
        )
    except Exception as e:
        logger.warning("query_reformulation_failed", error=str(e), using_original=True)
        reformulated_query = search_request.query

    # IMPROVEMENT: Adaptive hybrid search weighting based on query type
    query_type, semantic_weight, bm25_weight = classify_query(reformulated_query)
    logger.info(
        "query_classified",
        query_type=query_type.value,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight
    )

    # Query expansion (if enabled)
    expanded_queries = None
    queries_to_search = [reformulated_query]

    if search_request.expand_query:
        expansion_start = time.time()
        query_expansion_requests_total.inc()
        logger.debug("query_expansion_started", query=reformulated_query)

        expansion_service = get_expansion_service()
        expanded_queries = await expansion_service.expand_query(reformulated_query)
        queries_to_search = expanded_queries

        expansion_duration = time.time() - expansion_start
        query_expansion_duration_seconds.observe(expansion_duration)
        logger.info(
            "query_expansion_completed",
            num_queries=len(expanded_queries),
            duration_seconds=expansion_duration
        )

    # Perform multi-query search with improved retrieval
    all_result_lists: List[List[Dict]] = []  # Store separate lists for RRF
    from_cache = False

    for query_text in queries_to_search:
        # Embed query
        query_embeddings, provider_name = await router.embed(
            [query_text],
            required_dimension=collection.vector_dimension
        )
        query_embedding = query_embeddings[0]

        # Check cache if enabled (only for original query, not expanded)
        cache = get_cache()
        cached_results = None

        if search_request.use_cache and query_text == search_request.query:
            cached_results = await cache.get(str(collection_id), query_embedding)
            if cached_results:
                from_cache = True

        # Perform search if not cached
        if not from_cache:
            # IMPROVEMENT: Get 5x more candidates before reranking for better recall
            retrieval_multiplier = 5 if search_request.expand_query or search_request.hybrid else 3
            search_limit = search_request.limit * retrieval_multiplier

            search_results = await qdrant.search(
                collection_id=str(collection_id),
                query_vector=query_embedding,
                limit=search_limit,
                filters=search_request.filters
            )

            # Apply hybrid search if requested
            if search_request.hybrid and search_results:
                # IMPROVEMENT: Use adaptive weighting based on query type
                search_results = await hybrid_search(
                    query=query_text,
                    dense_results=search_results,
                    limit=search_limit,  # Keep more candidates for RRF
                    alpha=semantic_weight  # Adaptive weight based on query classification
                )

            all_result_lists.append(search_results)

            # Cache results (only original query)
            if search_request.use_cache and query_text == search_request.query:
                await cache.set(str(collection_id), query_embedding, search_results)
        else:
            all_result_lists.append(cached_results)

    # IMPROVEMENT: Use Reciprocal Rank Fusion for multi-query fusion
    if len(all_result_lists) > 1:
        merged_results = reciprocal_rank_fusion(all_result_lists, k=60)
        logger.debug(
            "rrf_fusion_applied",
            num_queries=len(all_result_lists),
            total_unique=len(merged_results)
        )
    else:
        merged_results = all_result_lists[0] if all_result_lists else []
        merged_results = sorted(merged_results, key=lambda x: x["score"], reverse=True)

    # IMPROVEMENT: Cross-encoder re-ranking for better relevance
    if settings.enable_reranking and merged_results:
        try:
            reranker = get_reranker()
            merged_results = await reranker.rerank(
                query=reformulated_query,
                results=merged_results,
                top_k=settings.reranker_top_k
            )
            logger.info(
                "reranking_applied",
                num_results=len(merged_results),
                top_k=settings.reranker_top_k
            )
        except Exception as e:
            logger.warning(
                "reranking_failed",
                error=str(e),
                using_original_ranking=True
            )

    # IMPROVEMENT: Apply MMR for diversity (reduce redundant results)
    if len(merged_results) > search_request.limit:
        merged_results = maximal_marginal_relevance(
            merged_results,
            limit=search_request.limit,
            lambda_param=0.7  # 70% relevance, 30% diversity
        )
    else:
        merged_results = merged_results[:search_request.limit]

    # IMPROVEMENT: Context expansion disabled for parent-child chunking
    # Parent content is already stored with each child chunk, so no need to fetch surrounding chunks
    # try:
    #     merged_results = await expand_context_windows(
    #         results=merged_results,
    #         db=db,
    #         qdrant=qdrant,
    #         collection_id=str(collection_id),
    #         window_size=1  # Include 1 chunk before and after
    #     )
    #     logger.debug("context_windows_expanded", num_results=len(merged_results))
    # except Exception as e:
    #     logger.warning("context_expansion_failed", error=str(e), using_original_results=True)

    # Format results
    results = [
        SearchResult(
            entity_id=hit["entity_id"],
            content=hit["content"],
            title=hit.get("title"),
            score=hit["score"],
            metadata=hit.get("metadata", {})
        )
        for hit in merged_results
    ]

    # Response synthesis (if enabled)
    synthesized_answer = None
    tokens_used = None
    synthesis_cost = None

    if search_request.synthesize and results:
        synthesis_start = time.time()
        logger.debug("response_synthesis_started")

        synthesis_service = get_synthesis_service()
        synthesized_answer, tokens_used = await synthesis_service.synthesize_answer(
            query=search_request.query,
            search_results=results,
            plan=None  # Will be used when adaptive RAG is enabled
        )
        # Ollama is free, so cost is 0
        synthesis_cost = 0.0

        synthesis_duration = time.time() - synthesis_start
        record_synthesis_metrics(synthesis_duration, tokens_used)
        logger.info(
            "response_synthesis_completed",
            tokens_used=tokens_used,
            duration_seconds=synthesis_duration
        )

    latency_ms = int((time.time() - start_time) * 1000)

    # Record search metrics
    record_search_metrics(
        collection_id=str(collection_id),
        duration_seconds=(time.time() - start_time),
        result_count=len(results),
        cache_hit=from_cache
    )

    logger.info(
        "search_completed",
        collection_id=str(collection_id),
        result_count=len(results),
        latency_ms=latency_ms,
        cache_hit=from_cache,
        expanded=search_request.expand_query,
        synthesized=search_request.synthesize
    )

    # Track analytics
    analytics = SearchAnalytics(
        collection_id=collection_id,
        query=search_request.query,
        result_count=len(results),
        latency_ms=latency_ms,
        cache_hit=from_cache,
        provider_used=provider_name,
        cost_usd=synthesis_cost or 0.0,
        search_metadata={
            "filters": search_request.filters,
            "expanded_queries": expanded_queries,
            "synthesized": search_request.synthesize
        } if (search_request.filters or expanded_queries or search_request.synthesize) else None
    )
    db.add(analytics)
    await db.commit()
    await db.refresh(analytics)  # Get the generated ID

    return SearchResponse(
        search_id=analytics.id,
        collection_id=collection_id,
        query=search_request.query,
        results=results,
        total_results=len(results),
        latency_ms=latency_ms,
        from_cache=from_cache,
        provider_used=provider_name,
        expanded_queries=expanded_queries,
        synthesized_answer=synthesized_answer,
        tokens_used=tokens_used,
        synthesis_cost_usd=synthesis_cost
    )
