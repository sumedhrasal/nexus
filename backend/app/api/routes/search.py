"""Search endpoints."""

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from app.models.database import Collection, SearchAnalytics
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.api.dependencies import get_db, get_qdrant_client, get_embedding_router
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.search.cache import get_cache

router = APIRouter(prefix="/collections/{collection_id}/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_collection(
    collection_id: uuid.UUID,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client),
    router: ProviderRouter = Depends(get_embedding_router)
):
    """Search a collection."""
    start_time = time.time()

    # Verify collection exists
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Embed query
    query_embeddings, provider_name = await router.embed(
        [request.query],
        required_dimension=collection.vector_dimension
    )
    query_embedding = query_embeddings[0]

    # Check cache if enabled
    cache = get_cache()
    cached_results = None
    from_cache = False

    if request.use_cache:
        cached_results = await cache.get(str(collection_id), query_embedding)
        if cached_results:
            from_cache = True

    # Perform search if not cached
    if not from_cache:
        search_results = await qdrant.search(
            collection_id=str(collection_id),
            query_vector=query_embedding,
            limit=request.limit,
            filters=request.filters
        )

        # Format results
        results = [
            SearchResult(
                entity_id=hit.payload["entity_id"],
                content=hit.payload["content"],
                title=hit.payload.get("title"),
                score=hit.score,
                metadata=hit.payload.get("metadata", {})
            )
            for hit in search_results
        ]

        # Cache results
        if request.use_cache:
            results_dict = [r.model_dump() for r in results]
            await cache.set(str(collection_id), query_embedding, results_dict)
    else:
        # Load cached results
        results = [SearchResult(**r) for r in cached_results]

    latency_ms = int((time.time() - start_time) * 1000)

    # Track analytics
    analytics = SearchAnalytics(
        collection_id=collection_id,
        query=request.query,
        result_count=len(results),
        latency_ms=latency_ms,
        cache_hit=from_cache,
        provider_used=provider_name,
        cost_usd=0.0,  # Cost tracking can be enhanced later
        search_metadata={"filters": request.filters} if request.filters else None
    )
    db.add(analytics)
    await db.commit()

    return SearchResponse(
        collection_id=collection_id,
        query=request.query,
        results=results,
        total_results=len(results),
        latency_ms=latency_ms,
        from_cache=from_cache,
        provider_used=provider_name
    )
