"""Semantic query caching with Redis."""

import json
import hashlib
from typing import Optional, List, Dict, Any
import redis.asyncio as redis
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache for search queries using Redis."""

    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: int = 3600):
        """Initialize semantic cache.

        Args:
            redis_url: Redis connection URL
            ttl_seconds: Cache TTL in seconds (default: 1 hour)
        """
        self.redis_url = redis_url or settings.redis_url
        self.ttl_seconds = ttl_seconds
        self.client: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis."""
        if not self.client:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            self.client = None

    def _make_key(self, collection_id: str, query_embedding: List[float]) -> str:
        """Create cache key from query embedding.

        Args:
            collection_id: Collection ID
            query_embedding: Query embedding vector

        Returns:
            Cache key string
        """
        # Hash the embedding for consistent key generation
        embedding_str = json.dumps(query_embedding)
        embedding_hash = hashlib.sha256(embedding_str.encode()).hexdigest()[:16]
        return f"search:{collection_id}:{embedding_hash}"

    async def get(
        self,
        collection_id: str,
        query_embedding: List[float]
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results.

        Args:
            collection_id: Collection ID
            query_embedding: Query embedding vector

        Returns:
            Cached results or None if not found
        """
        if not self.client:
            await self.connect()

        key = self._make_key(collection_id, query_embedding)

        try:
            cached = await self.client.get(key)
            if cached:
                logger.info(f"Cache hit for collection {collection_id}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")

        return None

    async def set(
        self,
        collection_id: str,
        query_embedding: List[float],
        results: List[Dict[str, Any]]
    ):
        """Cache search results.

        Args:
            collection_id: Collection ID
            query_embedding: Query embedding vector
            results: Search results to cache
        """
        if not self.client:
            await self.connect()

        key = self._make_key(collection_id, query_embedding)

        try:
            await self.client.setex(
                key,
                self.ttl_seconds,
                json.dumps(results)
            )
            logger.info(f"Cached results for collection {collection_id}")
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def invalidate_collection(self, collection_id: str):
        """Invalidate all cache entries for a collection.

        Args:
            collection_id: Collection ID
        """
        if not self.client:
            await self.connect()

        try:
            pattern = f"search:{collection_id}:*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            logger.info(f"Invalidated {deleted} cache entries for collection {collection_id}")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")


# Singleton instance
_cache_instance: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    """Get cache singleton instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
