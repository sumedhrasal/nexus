"""Search quality metrics endpoints."""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import Collection, SearchAnalytics, SearchFeedback
from app.models.schemas import (
    SearchFeedbackCreate,
    SearchFeedbackResponse,
    QualityMetrics
)
from app.api.dependencies import get_db
from app.search.quality_metrics import QualityMetricsService
from app.search.plan_metrics import get_plan_metrics
from app.search.plan_cache import get_plan_cache
from app.core.logging import get_logger

router = APIRouter(tags=["metrics"])
logger = get_logger(__name__)


@router.post("/search/{search_id}/feedback", response_model=SearchFeedbackResponse)
async def submit_search_feedback(
    search_id: uuid.UUID,
    feedback: SearchFeedbackCreate,
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback for a search result.

    This endpoint tracks user interactions with search results (clicks, copies, ratings)
    which are used to calculate quality metrics like Precision@k and MRR.

    Args:
        search_id: UUID of the search (from SearchAnalytics)
        feedback: Feedback data (position, entity_id, type)

    Returns:
        Created feedback record

    Raises:
        HTTPException: If search not found
    """
    # Verify search exists
    result = await db.execute(
        select(SearchAnalytics).where(SearchAnalytics.id == search_id)
    )
    search_analytics = result.scalar_one_or_none()

    if not search_analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Search {search_id} not found"
        )

    # Create feedback record
    db_feedback = SearchFeedback(
        search_analytics_id=search_id,
        result_position=feedback.result_position,
        result_entity_id=feedback.result_entity_id,
        feedback_type=feedback.feedback_type,
        feedback_metadata=feedback.feedback_metadata
    )
    db.add(db_feedback)
    await db.commit()
    await db.refresh(db_feedback)

    logger.info(
        "search_feedback_submitted",
        search_id=str(search_id),
        position=feedback.result_position,
        feedback_type=feedback.feedback_type
    )

    return db_feedback


@router.get("/collections/{collection_id}/quality-metrics", response_model=QualityMetrics)
async def get_quality_metrics(
    collection_id: uuid.UUID,
    period_days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """Get search quality metrics for a collection.

    Calculates metrics including:
    - Precision@k (k=1,3,5,10): Proportion of top-k results that were relevant
    - MRR (Mean Reciprocal Rank): Average reciprocal rank of first relevant result
    - Coverage: Percentage of searches that returned results
    - Click-through rate: Percentage of searches that received clicks

    Args:
        collection_id: Collection UUID
        period_days: Number of days to analyze (default: 30, max: 365)

    Returns:
        Quality metrics

    Raises:
        HTTPException: If collection not found
    """
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

    # Calculate metrics
    metrics = await QualityMetricsService.calculate_metrics(
        db=db,
        collection_id=collection_id,
        period_days=period_days
    )

    logger.info(
        "quality_metrics_calculated",
        collection_id=str(collection_id),
        period_days=period_days,
        total_searches=metrics["total_searches"],
        mrr=metrics["mrr"]
    )

    return QualityMetrics(**metrics)


@router.get("/plan-metrics")
async def get_plan_metrics_summary():
    """Get summary of execution plan metrics.

    Returns metrics about plan generation including:
    - Total plans generated
    - Complexity distribution (simple/moderate/complex/research)
    - Strategy distribution (direct/decompose/iterative)
    - Feature usage rates (decomposition, iterative retrieval)
    - Average confidence and low confidence rate
    - Fallback rate

    Returns:
        Plan metrics summary dictionary
    """
    metrics = get_plan_metrics()
    summary = metrics.get_summary()

    logger.info(
        "plan_metrics_requested",
        total_plans=summary["total_plans"],
        fallback_rate=summary["fallback_rate"]
    )

    return summary


@router.post("/plan-metrics/reset")
async def reset_plan_metrics():
    """Reset plan metrics counters.

    This endpoint resets all plan metrics to zero. Useful for starting
    fresh measurements after system changes or for periodic resets.

    Returns:
        Confirmation message
    """
    metrics = get_plan_metrics()
    metrics.reset()

    logger.info("plan_metrics_reset_requested")

    return {"status": "success", "message": "Plan metrics reset successfully"}


@router.get("/plan-cache/stats")
async def get_plan_cache_stats():
    """Get plan cache statistics.

    Returns cache hit/miss rates and configuration.

    Returns:
        Cache statistics including:
        - enabled: Whether caching is enabled
        - cache_hits: Number of cache hits
        - cache_misses: Number of cache misses
        - total_requests: Total cache requests
        - hit_rate: Cache hit rate (0-1)
        - ttl_seconds: TTL for cached plans
    """
    cache = await get_plan_cache()
    stats = cache.get_stats()

    logger.info(
        "plan_cache_stats_requested",
        hit_rate=stats["hit_rate"],
        total_requests=stats["total_requests"]
    )

    return stats


@router.post("/plan-cache/reset-stats")
async def reset_plan_cache_stats():
    """Reset plan cache statistics counters.

    Resets hit/miss counters without clearing cached plans.

    Returns:
        Confirmation message
    """
    cache = await get_plan_cache()
    cache.reset_stats()

    logger.info("plan_cache_stats_reset_requested")

    return {"status": "success", "message": "Plan cache statistics reset successfully"}


@router.post("/plan-cache/clear")
async def clear_plan_cache():
    """Clear all cached execution plans.

    This endpoint deletes all cached plans from Redis. Use this after
    configuration changes that affect plan generation.

    Returns:
        Number of plans deleted
    """
    cache = await get_plan_cache()
    deleted_count = await cache.clear_all()

    logger.info(
        "plan_cache_cleared_requested",
        plans_deleted=deleted_count
    )

    return {
        "status": "success",
        "message": f"Cleared {deleted_count} cached plans",
        "plans_deleted": deleted_count
    }
