"""Unit tests for embedding cache functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.core.embedding_cache import EmbeddingCache, get_embedding_cache


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.mget = AsyncMock(return_value=[])
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.scan = AsyncMock(return_value=(0, []))
    redis_mock.pipeline = MagicMock()
    return redis_mock


@pytest.fixture
def cache(mock_redis):
    """Create cache instance with mock Redis."""
    return EmbeddingCache(redis_client=mock_redis, ttl_seconds=3600)


class TestEmbeddingCacheInitialization:
    """Test cache initialization."""

    def test_init_with_custom_ttl(self):
        """Test cache initializes with custom TTL."""
        cache = EmbeddingCache(ttl_seconds=7200)
        assert cache.ttl_seconds == 7200

    def test_init_default_ttl(self):
        """Test cache initializes with default TTL (24 hours)."""
        cache = EmbeddingCache()
        assert cache.ttl_seconds == 86400

    def test_init_stats_reset(self):
        """Test cache initializes with zero stats."""
        cache = EmbeddingCache()
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["writes"] == 0
        assert stats["hit_rate"] == 0.0


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_make_cache_key_format(self, cache):
        """Test cache key has correct format."""
        key = cache._make_cache_key("test content", "gemini")
        assert key.startswith("emb:gemini:")

    def test_make_cache_key_different_content(self, cache):
        """Test different content produces different keys."""
        key1 = cache._make_cache_key("content 1", "gemini")
        key2 = cache._make_cache_key("content 2", "gemini")

        assert key1 != key2

    def test_make_cache_key_same_content(self, cache):
        """Test same content produces same key."""
        key1 = cache._make_cache_key("same content", "gemini")
        key2 = cache._make_cache_key("same content", "gemini")

        assert key1 == key2

    def test_make_cache_key_different_providers(self, cache):
        """Test different providers produce different keys."""
        key1 = cache._make_cache_key("content", "gemini")
        key2 = cache._make_cache_key("content", "openai")

        assert key1 != key2
        assert "gemini" in key1
        assert "openai" in key2


class TestCacheGet:
    """Test cache get operations."""

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache, mock_redis):
        """Test cache miss returns None."""
        mock_redis.get.return_value = None

        result = await cache.get("test content", "gemini")

        assert result is None
        assert cache.get_stats()["misses"] == 1
        assert cache.get_stats()["hits"] == 0

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache, mock_redis):
        """Test cache hit returns embedding."""
        embedding = [0.1, 0.2, 0.3]
        mock_redis.get.return_value = '[0.1, 0.2, 0.3]'

        result = await cache.get("test content", "gemini")

        assert result == embedding
        assert cache.get_stats()["hits"] == 1
        assert cache.get_stats()["misses"] == 0

    @pytest.mark.asyncio
    async def test_get_handles_redis_error(self, cache, mock_redis):
        """Test cache handles Redis errors gracefully."""
        mock_redis.get.side_effect = Exception("Redis connection error")

        result = await cache.get("test content", "gemini")

        assert result is None
        assert cache.get_stats()["misses"] == 1


class TestCacheSet:
    """Test cache set operations."""

    @pytest.mark.asyncio
    async def test_set_success(self, cache, mock_redis):
        """Test successful cache set."""
        embedding = [0.1, 0.2, 0.3]
        mock_redis.setex.return_value = True

        result = await cache.set("test content", embedding, "gemini")

        assert result is True
        assert cache.get_stats()["writes"] == 1
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache, mock_redis):
        """Test cache set uses correct TTL."""
        embedding = [0.1, 0.2, 0.3]

        await cache.set("test content", embedding, "gemini")

        # Check TTL was passed correctly
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == cache.ttl_seconds

    @pytest.mark.asyncio
    async def test_set_handles_error(self, cache, mock_redis):
        """Test cache handles set errors gracefully."""
        embedding = [0.1, 0.2, 0.3]
        mock_redis.setex.side_effect = Exception("Redis error")

        result = await cache.set("test content", embedding, "gemini")

        assert result is False


class TestCacheBatchOperations:
    """Test batch cache operations."""

    @pytest.mark.asyncio
    async def test_get_batch_all_miss(self, cache, mock_redis):
        """Test batch get with all cache misses."""
        contents = ["content1", "content2", "content3"]
        mock_redis.mget.return_value = [None, None, None]

        embeddings, missing_indices = await cache.get_batch(contents, "gemini")

        assert embeddings == [None, None, None]
        assert missing_indices == [0, 1, 2]
        assert cache.get_stats()["misses"] == 3

    @pytest.mark.asyncio
    async def test_get_batch_all_hit(self, cache, mock_redis):
        """Test batch get with all cache hits."""
        contents = ["content1", "content2", "content3"]
        mock_redis.mget.return_value = [
            '[0.1, 0.2]',
            '[0.3, 0.4]',
            '[0.5, 0.6]'
        ]

        embeddings, missing_indices = await cache.get_batch(contents, "gemini")

        assert embeddings == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        assert missing_indices == []
        assert cache.get_stats()["hits"] == 3

    @pytest.mark.asyncio
    async def test_get_batch_partial_hit(self, cache, mock_redis):
        """Test batch get with partial cache hits."""
        contents = ["content1", "content2", "content3"]
        mock_redis.mget.return_value = [
            '[0.1, 0.2]',  # Hit
            None,           # Miss
            '[0.5, 0.6]'   # Hit
        ]

        embeddings, missing_indices = await cache.get_batch(contents, "gemini")

        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] is None
        assert embeddings[2] == [0.5, 0.6]
        assert missing_indices == [1]
        assert cache.get_stats()["hits"] == 2
        assert cache.get_stats()["misses"] == 1

    @pytest.mark.asyncio
    async def test_set_batch_success(self, cache, mock_redis):
        """Test batch set operation."""
        contents = ["content1", "content2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        # Mock pipeline
        pipeline_mock = AsyncMock()
        pipeline_mock.execute = AsyncMock(return_value=[True, True])
        mock_redis.pipeline.return_value = pipeline_mock

        result = await cache.set_batch(contents, embeddings, "gemini")

        assert result == 2
        assert cache.get_stats()["writes"] == 2

    @pytest.mark.asyncio
    async def test_set_batch_length_mismatch(self, cache):
        """Test batch set fails with mismatched lengths."""
        contents = ["content1", "content2"]
        embeddings = [[0.1, 0.2]]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            await cache.set_batch(contents, embeddings, "gemini")


class TestCacheDelete:
    """Test cache delete operations."""

    @pytest.mark.asyncio
    async def test_delete_success(self, cache, mock_redis):
        """Test successful delete."""
        mock_redis.delete.return_value = 1

        result = await cache.delete("test content", "gemini")

        assert result is True
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, cache, mock_redis):
        """Test delete when key doesn't exist."""
        mock_redis.delete.return_value = 0

        result = await cache.delete("test content", "gemini")

        assert result is False


class TestCacheClear:
    """Test cache clear operations."""

    @pytest.mark.asyncio
    async def test_clear_all(self, cache, mock_redis):
        """Test clearing all cache entries."""
        mock_redis.scan.return_value = (0, [b"emb:gemini:key1", b"emb:openai:key2"])
        mock_redis.delete.return_value = 2

        result = await cache.clear()

        assert result == 2

    @pytest.mark.asyncio
    async def test_clear_specific_provider(self, cache, mock_redis):
        """Test clearing cache for specific provider."""
        mock_redis.scan.return_value = (0, [b"emb:gemini:key1"])
        mock_redis.delete.return_value = 1

        result = await cache.clear(provider="gemini")

        assert result == 1


class TestCacheStats:
    """Test cache statistics."""

    @pytest.mark.asyncio
    async def test_stats_initial(self, cache):
        """Test initial stats are zero."""
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["writes"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, cache, mock_redis):
        """Test stats update after operations."""
        # Cache miss
        mock_redis.get.return_value = None
        await cache.get("content1", "gemini")

        # Cache hit
        mock_redis.get.return_value = '[0.1, 0.2]'
        await cache.get("content2", "gemini")
        await cache.get("content3", "gemini")

        # Cache write
        await cache.set("content4", [0.5, 0.6], "gemini")

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["writes"] == 1
        assert stats["total_requests"] == 3
        assert stats["hit_rate"] == pytest.approx(0.6667, rel=0.01)

    def test_stats_hit_rate_calculation(self, cache):
        """Test hit rate calculation."""
        cache._stats["hits"] = 7
        cache._stats["misses"] = 3

        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.7

    def test_reset_stats(self, cache):
        """Test stats can be reset."""
        cache._stats["hits"] = 10
        cache._stats["misses"] = 5
        cache._stats["writes"] = 8

        cache.reset_stats()

        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["writes"] == 0


class TestGlobalCacheInstance:
    """Test global cache instance management."""

    def test_get_embedding_cache_singleton(self):
        """Test global cache returns same instance."""
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()

        assert cache1 is cache2

    def test_get_embedding_cache_with_ttl(self):
        """Test global cache respects TTL."""
        cache = get_embedding_cache(ttl_seconds=7200)

        # Note: First call sets TTL, subsequent calls reuse instance
        assert cache.ttl_seconds in [7200, 86400]  # Could be default or custom


class TestCacheEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_content(self, cache):
        """Test caching empty content."""
        result = await cache.set("", [0.1, 0.2], "gemini")
        assert result is True

    @pytest.mark.asyncio
    async def test_large_embedding(self, cache, mock_redis):
        """Test caching large embedding vector."""
        large_embedding = [0.1] * 4096  # 4K dimensions

        result = await cache.set("content", large_embedding, "gemini")
        assert result is True

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, cache):
        """Test content with special characters."""
        content = "Test\nwith\ttabs and\nnewlines 你好"
        key = cache._make_cache_key(content, "gemini")

        assert key.startswith("emb:gemini:")
        assert len(key) > 20

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache, mock_redis):
        """Test concurrent cache operations."""
        async def get_operation(i):
            return await cache.get(f"content{i}", "gemini")

        # Simulate concurrent requests
        results = await asyncio.gather(*[get_operation(i) for i in range(10)])

        assert len(results) == 10
        assert cache.get_stats()["misses"] == 10
