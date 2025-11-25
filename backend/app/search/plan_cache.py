"""Plan caching for adaptive RAG to reduce redundant LLM planning calls.

This module provides Redis-backed caching for execution plans, reducing
latency and API costs for similar or repeated queries.
"""

import hashlib
import json
from typing import Optional
from redis.asyncio import Redis
from app.search.plan_schema import ExecutionPlan
from app.core.logging import get_logger

logger = get_logger(__name__)


class PlanCache:
    """Caches execution plans to avoid redundant LLM planning calls."""

    def __init__(
        self,
        redis_client: Redis,
        ttl_seconds: int = 3600,
        enabled: bool = True
    ):
        """Initialize plan cache.

        Args:
            redis_client: Redis client for cache storage
            ttl_seconds: Time-to-live for cached plans (default: 1 hour)
            enabled: Whether caching is enabled
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.enabled = enabled

        # Metrics
        self.cache_hits = 0
        self.cache_misses = 0

    async def get(self, query: str) -> Optional[ExecutionPlan]:
        """Get cached execution plan for query.

        Args:
            query: User query

        Returns:
            Cached ExecutionPlan if found, None otherwise
        """
        if not self.enabled:
            return None

        try:
            cache_key = self._generate_cache_key(query)
            redis_key = f"plan:{cache_key}"

            cached_data = await self.redis.get(redis_key)

            if cached_data:
                # Parse and reconstruct ExecutionPlan
                plan_dict = json.loads(cached_data)
                plan = ExecutionPlan(**plan_dict)

                self.cache_hits += 1

                logger.info(
                    "plan_cache_hit",
                    query=query[:100],
                    cache_key=cache_key,
                    strategy=plan.strategy.value,
                    complexity=plan.complexity.value
                )

                return plan

            self.cache_misses += 1
            logger.debug("plan_cache_miss", query=query[:100], cache_key=cache_key)
            return None

        except Exception as e:
            logger.error(
                "plan_cache_get_failed",
                error=str(e),
                query=query[:100],
                exc_info=True
            )
            # Don't fail on cache errors, just return None
            return None

    async def set(self, query: str, plan: ExecutionPlan) -> bool:
        """Cache execution plan for query.

        Args:
            query: User query
            plan: Execution plan to cache

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            cache_key = self._generate_cache_key(query)
            redis_key = f"plan:{cache_key}"

            # Serialize plan to JSON
            plan_dict = plan.model_dump()  # Pydantic v2
            plan_json = json.dumps(plan_dict)

            # Store with TTL
            await self.redis.setex(redis_key, self.ttl, plan_json)

            logger.debug(
                "plan_cached",
                query=query[:100],
                cache_key=cache_key,
                ttl_seconds=self.ttl,
                strategy=plan.strategy.value
            )

            return True

        except Exception as e:
            logger.error(
                "plan_cache_set_failed",
                error=str(e),
                query=query[:100],
                exc_info=True
            )
            # Don't fail on cache errors
            return False

    async def delete(self, query: str) -> bool:
        """Delete cached plan for query.

        Args:
            query: User query

        Returns:
            True if deleted, False if not found or error
        """
        if not self.enabled:
            return False

        try:
            cache_key = self._generate_cache_key(query)
            redis_key = f"plan:{cache_key}"

            deleted = await self.redis.delete(redis_key)

            logger.debug(
                "plan_cache_deleted",
                query=query[:100],
                cache_key=cache_key,
                deleted=bool(deleted)
            )

            return bool(deleted)

        except Exception as e:
            logger.error(
                "plan_cache_delete_failed",
                error=str(e),
                query=query[:100],
                exc_info=True
            )
            return False

    async def clear_all(self) -> int:
        """Clear all cached plans.

        Returns:
            Number of plans deleted
        """
        if not self.enabled:
            return 0

        try:
            # Find all plan keys
            pattern = "plan:*"
            keys = []

            # Scan for keys (more efficient than KEYS for large datasets)
            cursor = 0
            while True:
                cursor, batch = await self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys.extend(batch)

                if cursor == 0:
                    break

            # Delete all keys
            if keys:
                deleted = await self.redis.delete(*keys)
            else:
                deleted = 0

            logger.info(
                "plan_cache_cleared",
                keys_deleted=deleted
            )

            return deleted

        except Exception as e:
            logger.error(
                "plan_cache_clear_failed",
                error=str(e),
                exc_info=True
            )
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        total = self.cache_hits + self.cache_misses

        return {
            "enabled": self.enabled,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total,
            "hit_rate": (self.cache_hits / total) if total > 0 else 0.0,
            "ttl_seconds": self.ttl
        }

    def reset_stats(self):
        """Reset cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("plan_cache_stats_reset")

    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key from query.

        Normalizes query and generates MD5 hash for consistent keys.

        Args:
            query: User query

        Returns:
            Cache key (MD5 hash)
        """
        # Normalize query:
        # 1. Convert to lowercase
        # 2. Strip whitespace
        # 3. Collapse multiple spaces to single space
        normalized = " ".join(query.lower().strip().split())

        # Generate MD5 hash
        hash_obj = hashlib.md5(normalized.encode('utf-8'))
        cache_key = hash_obj.hexdigest()

        return cache_key


# Singleton instance
_plan_cache: Optional[PlanCache] = None


async def get_plan_cache(
    redis_client: Optional[Redis] = None,
    ttl_seconds: Optional[int] = None,
    enabled: Optional[bool] = None
) -> PlanCache:
    """Get or create plan cache singleton.

    Args:
        redis_client: Optional Redis client override
        ttl_seconds: Optional TTL override
        enabled: Optional enabled flag override

    Returns:
        PlanCache instance
    """
    global _plan_cache

    # Import here to avoid circular dependency
    from app.config import settings
    from app.search.cache import get_cache

    # Use provided values or defaults from settings
    if ttl_seconds is None:
        ttl_seconds = getattr(settings, 'plan_cache_ttl_seconds', 3600)

    if enabled is None:
        enabled = getattr(settings, 'enable_plan_caching', True)

    if _plan_cache is None:
        # Get Redis client from existing cache system
        if redis_client is None:
            cache = get_cache()
            redis_client = cache.redis

        _plan_cache = PlanCache(
            redis_client=redis_client,
            ttl_seconds=ttl_seconds,
            enabled=enabled
        )

    return _plan_cache
