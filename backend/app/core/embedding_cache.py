"""Embedding cache using Redis for cost savings and performance."""

import json
import logging
from typing import List, Optional, Tuple
import redis.asyncio as redis
from app.config import settings
from app.core.hashing import hash_content

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Redis-based cache for embeddings."""

    def __init__(self, redis_client: Optional[redis.Redis] = None, ttl_seconds: int = 86400):
        """Initialize embedding cache.

        Args:
            redis_client: Redis client instance (creates one if not provided)
            ttl_seconds: Time-to-live for cache entries (default: 24 hours)
        """
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self._stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0
        }

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis client."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=False,  # We'll handle encoding
                encoding="utf-8"
            )
        return self.redis_client

    def _make_cache_key(self, content: str, provider: str = "default") -> str:
        """Generate cache key from content and provider.

        Args:
            content: Text content to embed
            provider: Provider name (e.g., "gemini", "openai")

        Returns:
            Cache key in format "emb:{provider}:{content_hash}"
        """
        content_hash = hash_content(content)
        return f"emb:{provider}:{content_hash}"

    async def get(self, content: str, provider: str = "default") -> Optional[List[float]]:
        """Get embedding from cache.

        Args:
            content: Text content
            provider: Provider name

        Returns:
            Embedding vector if found, None otherwise
        """
        cache_key = self._make_cache_key(content, provider)
        redis_client = await self._get_redis()

        try:
            cached_value = await redis_client.get(cache_key)

            if cached_value:
                self._stats["hits"] += 1
                embedding = json.loads(cached_value)
                logger.debug(f"Cache HIT for key: {cache_key}")
                return embedding
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache MISS for key: {cache_key}")
                return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self._stats["misses"] += 1
            return None

    async def get_batch(
        self,
        contents: List[str],
        provider: str = "default"
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """Get multiple embeddings from cache.

        Args:
            contents: List of text contents
            provider: Provider name

        Returns:
            Tuple of (embeddings_list, missing_indices)
            - embeddings_list: List where cached embeddings are filled in, None for misses
            - missing_indices: Indices of contents that need to be embedded
        """
        cache_keys = [self._make_cache_key(content, provider) for content in contents]
        redis_client = await self._get_redis()

        embeddings: List[Optional[List[float]]] = [None] * len(contents)
        missing_indices: List[int] = []

        try:
            # Batch get from Redis
            cached_values = await redis_client.mget(cache_keys)

            for i, cached_value in enumerate(cached_values):
                if cached_value:
                    embeddings[i] = json.loads(cached_value)
                    self._stats["hits"] += 1
                    logger.debug(f"Cache HIT for index {i}")
                else:
                    missing_indices.append(i)
                    self._stats["misses"] += 1
                    logger.debug(f"Cache MISS for index {i}")

        except Exception as e:
            logger.error(f"Error getting batch from cache: {e}")
            # On error, treat all as misses
            missing_indices = list(range(len(contents)))
            self._stats["misses"] += len(contents)

        return embeddings, missing_indices

    async def set(
        self,
        content: str,
        embedding: List[float],
        provider: str = "default"
    ) -> bool:
        """Store embedding in cache.

        Args:
            content: Text content
            embedding: Embedding vector
            provider: Provider name

        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = self._make_cache_key(content, provider)
        redis_client = await self._get_redis()

        try:
            # Serialize embedding to JSON
            embedding_json = json.dumps(embedding)

            # Store with TTL
            await redis_client.setex(
                cache_key,
                self.ttl_seconds,
                embedding_json
            )

            self._stats["writes"] += 1
            logger.debug(f"Cached embedding for key: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def set_batch(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        provider: str = "default"
    ) -> int:
        """Store multiple embeddings in cache.

        Args:
            contents: List of text contents
            embeddings: List of embedding vectors
            provider: Provider name

        Returns:
            Number of successfully cached embeddings
        """
        if len(contents) != len(embeddings):
            raise ValueError("Contents and embeddings lists must have same length")

        redis_client = await self._get_redis()
        success_count = 0

        try:
            # Prepare pipeline for batch operation
            pipe = redis_client.pipeline()

            for content, embedding in zip(contents, embeddings):
                cache_key = self._make_cache_key(content, provider)
                embedding_json = json.dumps(embedding)
                pipe.setex(cache_key, self.ttl_seconds, embedding_json)

            # Execute pipeline
            await pipe.execute()
            success_count = len(contents)
            self._stats["writes"] += success_count
            logger.debug(f"Batch cached {success_count} embeddings")

        except Exception as e:
            logger.error(f"Error batch setting cache: {e}")

        return success_count

    async def delete(self, content: str, provider: str = "default") -> bool:
        """Delete embedding from cache.

        Args:
            content: Text content
            provider: Provider name

        Returns:
            True if deleted, False otherwise
        """
        cache_key = self._make_cache_key(content, provider)
        redis_client = await self._get_redis()

        try:
            deleted = await redis_client.delete(cache_key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def clear(self, provider: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            provider: If specified, only clear entries for this provider.
                     If None, clears all embedding cache entries.

        Returns:
            Number of deleted keys
        """
        redis_client = await self._get_redis()

        try:
            if provider:
                pattern = f"emb:{provider}:*"
            else:
                pattern = "emb:*"

            deleted = 0
            cursor = 0

            # Scan and delete in batches
            while True:
                cursor, keys = await redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                if keys:
                    deleted += await redis_client.delete(*keys)

                if cursor == 0:
                    break

            logger.info(f"Cleared {deleted} cache entries with pattern: {pattern}")
            return deleted

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, writes, and hit rate
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "writes": self._stats["writes"],
            "hit_rate": round(hit_rate, 4),
            "total_requests": total_requests
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0
        }


# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(ttl_seconds: int = 86400) -> EmbeddingCache:
    """Get the global embedding cache instance.

    Args:
        ttl_seconds: Time-to-live for cache entries (default: 24 hours)

    Returns:
        EmbeddingCache instance
    """
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(ttl_seconds=ttl_seconds)
    return _embedding_cache
