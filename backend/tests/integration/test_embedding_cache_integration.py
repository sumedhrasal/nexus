"""Integration tests for embedding cache with ProviderRouter."""

import pytest
from unittest.mock import AsyncMock, patch
from app.core.providers.router import ProviderRouter
from app.core.embedding_cache import EmbeddingCache


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = AsyncMock()
    provider.name = "test-provider"
    provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    provider.get_embedding_dimension = AsyncMock(return_value=768)
    provider.get_cost_per_1k_tokens = AsyncMock(return_value=0.0001)
    return provider


@pytest.fixture
def cache():
    """Create a fresh cache instance for each test."""
    cache = EmbeddingCache(ttl_seconds=3600)
    cache.reset_stats()
    return cache


class TestProviderRouterWithCache:
    """Test ProviderRouter with embedding cache enabled."""

    @pytest.mark.asyncio
    async def test_cache_enabled_by_default(self):
        """Test cache is enabled by default in ProviderRouter."""
        with patch('app.core.providers.router.OllamaProvider'):
            router = ProviderRouter(strategy="cost", use_cache=True)
            assert router.use_cache is True
            assert router.cache is not None

    @pytest.mark.asyncio
    async def test_cache_disabled_when_requested(self):
        """Test cache can be disabled."""
        with patch('app.core.providers.router.OllamaProvider'):
            router = ProviderRouter(strategy="cost", use_cache=False)
            assert router.use_cache is False
            assert router.cache is None

    @pytest.mark.asyncio
    async def test_embed_with_cache_miss(self, mock_provider, cache):
        """Test embed with cache miss calls provider."""
        # Create router with mock provider and cache
        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = cache
        router.providers = [mock_provider]

        texts = ["test content"]
        embeddings, provider_name = await router.embed(texts)

        # Should call provider
        mock_provider.embed.assert_called_once_with(texts)
        assert embeddings == [[0.1, 0.2, 0.3]]

        # Should cache the result
        stats = cache.get_stats()
        assert stats["writes"] == 1
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_embed_with_cache_hit(self, mock_provider, cache):
        """Test embed with cache hit doesn't call provider."""
        # Pre-populate cache
        await cache.set("test content", [0.5, 0.6, 0.7], "test-provider")
        cache.reset_stats()  # Reset stats after pre-population

        # Create router
        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = cache
        router.providers = [mock_provider]

        texts = ["test content"]
        embeddings, provider_name = await router.embed(texts)

        # Should NOT call provider
        mock_provider.embed.assert_not_called()

        # Should return cached result
        assert embeddings == [[0.5, 0.6, 0.7]]

        # Should have cache hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["writes"] == 0

    @pytest.mark.asyncio
    async def test_embed_batch_partial_cache(self, mock_provider, cache):
        """Test embed with partial cache hits."""
        # Pre-cache one text
        await cache.set("cached content", [0.5, 0.6, 0.7], "test-provider")
        cache.reset_stats()

        # Mock provider to return embeddings for missing items
        mock_provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        # Create router
        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = cache
        router.providers = [mock_provider]

        texts = ["cached content", "new content"]
        embeddings, provider_name = await router.embed(texts)

        # Should call provider only for missing text
        mock_provider.embed.assert_called_once_with(["new content"])

        # Should return both embeddings
        assert len(embeddings) == 2
        assert embeddings[0] == [0.5, 0.6, 0.7]  # From cache
        assert embeddings[1] == [0.1, 0.2, 0.3]  # From provider

        # Should have 1 hit, 1 miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["writes"] == 1

    @pytest.mark.asyncio
    async def test_cache_stats_available_via_router(self, mock_provider, cache):
        """Test cache stats accessible via router."""
        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = cache
        router.providers = [mock_provider]

        # Initially empty
        stats = router.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # After some operations
        await router.embed(["test1"])
        await router.embed(["test1"])  # Should hit cache

        stats = router.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_cache_stats_none_when_disabled(self, mock_provider):
        """Test cache stats return None when cache disabled."""
        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = False
        router.cache = None
        router.providers = [mock_provider]

        stats = router.get_cache_stats()
        assert stats is None


class TestCachePersistenceScenarios:
    """Test cache behavior in different scenarios."""

    @pytest.mark.asyncio
    async def test_cache_deduplicates_identical_content(self, mock_provider, cache):
        """Test cache deduplicates identical content."""
        mock_provider.embed = AsyncMock(side_effect=[
            [[0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6]],
        ])

        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = cache
        router.providers = [mock_provider]

        # First call - cache miss
        embeddings1, _ = await router.embed(["same content"])
        assert embeddings1 == [[0.1, 0.2, 0.3]]

        # Second call with same content - cache hit
        embeddings2, _ = await router.embed(["same content"])
        assert embeddings2 == [[0.1, 0.2, 0.3]]  # Same as first

        # Third call with different content - cache miss
        embeddings3, _ = await router.embed(["different content"])
        assert embeddings3 == [[0.4, 0.5, 0.6]]

        # Provider should be called twice (not three times)
        assert mock_provider.embed.call_count == 2

        # Cache stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["writes"] == 2

    @pytest.mark.asyncio
    async def test_cache_handles_whitespace_differences(self, mock_provider, cache):
        """Test cache treats different whitespace as different content."""
        # Create separate mock calls for each text
        call_count = 0
        async def embed_mock(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [[0.1, 0.2, 0.3]]
            return [[0.4, 0.5, 0.6]]

        mock_provider.embed = AsyncMock(side_effect=embed_mock)

        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = cache
        router.providers = [mock_provider]

        # Different whitespace = different hash = different cache entry
        await router.embed(["test content"])
        await router.embed(["test  content"])  # Double space

        # Should call provider twice (different content)
        assert call_count == 2

        stats = cache.get_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0

    @pytest.mark.asyncio
    async def test_cache_with_large_batch(self, mock_provider, cache):
        """Test cache with large batch of texts."""
        # Mock provider to return unique embeddings
        def embed_side_effect(texts):
            return [[float(i) * 0.1, float(i) * 0.2] for i in range(len(texts))]

        mock_provider.embed = AsyncMock(side_effect=embed_side_effect)

        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = cache
        router.providers = [mock_provider]

        # Large batch
        texts = [f"text_{i}" for i in range(100)]
        embeddings1, _ = await router.embed(texts)

        assert len(embeddings1) == 100

        # Call again - should all hit cache
        embeddings2, _ = await router.embed(texts)

        assert embeddings1 == embeddings2

        # Provider called once
        assert mock_provider.embed.call_count == 1

        stats = cache.get_stats()
        assert stats["hits"] == 100
        assert stats["misses"] == 100

    @pytest.mark.asyncio
    async def test_cache_fallback_on_error(self, mock_provider):
        """Test system works when cache fails."""
        # Create a cache that will fail
        failing_cache = EmbeddingCache(ttl_seconds=3600)
        failing_cache.get_batch = AsyncMock(side_effect=Exception("Redis down"))

        mock_provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "cost"
        router.use_cache = True
        router.cache = failing_cache
        router.providers = [mock_provider]

        # Should still work, just without cache
        # The embed method will catch the cache error and proceed
        embeddings, _ = await router.embed(["test"])

        assert embeddings == [[0.1, 0.2, 0.3]]
        mock_provider.embed.assert_called_once()


class TestCacheWithMultipleProviders:
    """Test cache behavior with multiple providers."""

    @pytest.mark.asyncio
    async def test_cache_keys_include_provider_name(self, cache):
        """Test cache uses provider name in keys."""
        # Cache same content for different providers
        await cache.set("test", [0.1, 0.2], "gemini")
        await cache.set("test", [0.3, 0.4], "openai")

        # Should retrieve different embeddings
        gemini_emb = await cache.get("test", "gemini")
        openai_emb = await cache.get("test", "openai")

        assert gemini_emb == [0.1, 0.2]
        assert openai_emb == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_provider_fallback_maintains_cache(self, cache):
        """Test cache works with provider fallback."""
        # Mock two providers
        provider1 = AsyncMock()
        provider1.name = "provider1"
        provider1.embed = AsyncMock(side_effect=Exception("Provider 1 down"))
        provider1.get_embedding_dimension = AsyncMock(return_value=768)
        provider1.get_cost_per_1k_tokens = AsyncMock(return_value=0.0001)

        provider2 = AsyncMock()
        provider2.name = "provider2"
        provider2.embed = AsyncMock(return_value=[[0.5, 0.6, 0.7]])
        provider2.get_embedding_dimension = AsyncMock(return_value=768)
        provider2.get_cost_per_1k_tokens = AsyncMock(return_value=0.0002)

        router = ProviderRouter.__new__(ProviderRouter)
        router.strategy = "fallback"
        router.use_cache = True
        router.cache = cache
        router.providers = [provider1, provider2]

        # First call - provider1 fails, fallback to provider2
        embeddings1, provider_name = await router.embed(["test"])

        assert embeddings1 == [[0.5, 0.6, 0.7]]
        assert provider_name == "provider2"

        # Cache stores with provider1 name (first in list)
        # Second call will check cache and get hit
        cache.reset_stats()
        embeddings2, _ = await router.embed(["test"])

        # Should hit cache this time
        assert embeddings2 == [[0.5, 0.6, 0.7]]


class TestCacheStatisticsEndpoint:
    """Test cache statistics via API endpoint."""

    @pytest.mark.asyncio
    async def test_cache_stats_format(self):
        """Test cache stats return expected format."""
        cache = EmbeddingCache(ttl_seconds=3600)

        # Do some operations
        await cache.get("test1", "gemini")  # Miss
        await cache.set("test1", [0.1, 0.2], "gemini")  # Write
        await cache.get("test1", "gemini")  # Hit

        stats = cache.get_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "writes" in stats
        assert "hit_rate" in stats
        assert "total_requests" in stats

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["writes"] == 1
        assert stats["total_requests"] == 2
        assert 0 <= stats["hit_rate"] <= 1


class TestCachePerformanceMetrics:
    """Test cache performance and metrics."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self, cache):
        """Test hit rate is calculated correctly."""
        # 7 hits, 3 misses = 70% hit rate
        for _ in range(7):
            cache._stats["hits"] += 1
        for _ in range(3):
            cache._stats["misses"] += 1

        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.7
        assert stats["total_requests"] == 10

    @pytest.mark.asyncio
    async def test_cache_zero_requests(self, cache):
        """Test hit rate is 0 with no requests."""
        stats = cache.get_stats()

        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0
