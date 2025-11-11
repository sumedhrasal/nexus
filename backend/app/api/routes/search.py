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
from app.search.synthesis import get_synthesis_service
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
    qdrant: QdrantStorage = Depends(get_qdrant_client),
    router: ProviderRouter = Depends(get_embedding_router)
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

    # Query expansion (if enabled)
    expanded_queries = None
    queries_to_search = [search_request.query]

    if search_request.expand_query:
        expansion_start = time.time()
        query_expansion_requests_total.inc()
        logger.debug("query_expansion_started", query=search_request.query)

        expansion_service = get_expansion_service()
        expanded_queries = await expansion_service.expand_query(search_request.query)
        queries_to_search = expanded_queries

        expansion_duration = time.time() - expansion_start
        query_expansion_duration_seconds.observe(expansion_duration)
        logger.info(
            "query_expansion_completed",
            num_queries=len(expanded_queries),
            duration_seconds=expansion_duration
        )

    # Perform multi-query search
    all_results_map: Dict[str, Dict] = {}  # entity_id -> result dict (for deduplication)

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
        from_cache = False

        if search_request.use_cache and query_text == search_request.query:
            cached_results = await cache.get(str(collection_id), query_embedding)
            if cached_results:
                from_cache = True

        # Perform search if not cached
        if not from_cache:
            # Get more results for hybrid search reranking
            search_limit = search_request.limit * 3 if search_request.hybrid else search_request.limit

            search_results = await qdrant.search(
                collection_id=str(collection_id),
                query_vector=query_embedding,
                limit=search_limit,
                filters=search_request.filters
            )

            # Apply hybrid search if requested
            if search_request.hybrid and search_results:
                search_results = await hybrid_search(
                    query=query_text,
                    dense_results=search_results,
                    limit=search_request.limit
                )
            else:
                # Limit to requested amount
                search_results = search_results[:search_request.limit]

            # Merge results (deduplicate by entity_id, keep highest score)
            for hit in search_results:
                entity_id = hit["entity_id"]
                if entity_id not in all_results_map or hit["score"] > all_results_map[entity_id]["score"]:
                    all_results_map[entity_id] = hit

            # Cache results (only original query)
            if search_request.use_cache and query_text == search_request.query:
                results_dict = [r for r in search_results]
                await cache.set(str(collection_id), query_embedding, results_dict)
        else:
            # Load cached results
            for cached_result in cached_results:
                entity_id = cached_result["entity_id"]
                if entity_id not in all_results_map or cached_result["score"] > all_results_map[entity_id]["score"]:
                    all_results_map[entity_id] = cached_result

    # Convert to sorted list and limit to requested amount
    merged_results = sorted(all_results_map.values(), key=lambda x: x["score"], reverse=True)
    merged_results = merged_results[:search_request.limit]

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
            search_results=results
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

    return SearchResponse(
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
