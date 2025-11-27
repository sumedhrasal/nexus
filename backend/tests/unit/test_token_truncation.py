"""Tests for token splitting in embedding provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.providers.router import ProviderRouter
from app.core.token_counter import estimate_tokens, split_text_by_tokens


def test_estimate_tokens():
    """Test token estimation."""
    # Short text
    short = "Hello world"
    assert estimate_tokens(short) < 10

    # Long text (roughly 1000 tokens)
    long = " ".join(["word"] * 750)  # ~1000 tokens
    estimated = estimate_tokens(long)
    assert 900 < estimated < 1100  # Allow 10% margin


def test_split_text_by_tokens():
    """Test text splitting into chunks."""
    # Text that's already short enough
    short = "This is a short text"
    result = split_text_by_tokens(short, max_tokens=100)
    assert len(result) == 1
    assert result[0] == short

    # Text that needs splitting
    long = " ".join(["word"] * 1000)  # ~1333 tokens
    chunks = split_text_by_tokens(long, max_tokens=500, overlap_tokens=50)

    # Verify multiple chunks were created
    assert len(chunks) > 1

    # Verify each chunk is under limit
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 500

    # Verify all words are preserved (accounting for overlap)
    all_words = long.split()
    reconstructed_words = []
    for chunk in chunks:
        reconstructed_words.extend(chunk.split())

    # Should have at least as many words as original (due to overlap)
    assert len(reconstructed_words) >= len(all_words)


@pytest.mark.asyncio
async def test_router_splits_long_texts():
    """Test that router splits texts exceeding token limit into chunks."""
    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.name = "test_provider"
    mock_provider.get_embedding_dimension.return_value = 768
    mock_provider.get_max_embedding_tokens.return_value = 100  # Small limit for testing

    # Mock the embed method - returns different embeddings for each chunk
    call_count = [0]
    async def mock_embed(texts):
        embeddings = []
        for i, text in enumerate(texts):
            # Create slightly different embeddings for each chunk
            base_value = 0.1 + (call_count[0] + i) * 0.01
            embeddings.append([base_value] * 768)
        call_count[0] += len(texts)
        return embeddings

    mock_provider.embed = AsyncMock(side_effect=mock_embed)

    # Create router with mock provider
    router = ProviderRouter(strategy="local-first", use_cache=False)
    router.providers = [mock_provider]

    # Create a long text that exceeds 100 tokens
    long_text = " ".join(["word"] * 200)  # ~266 tokens
    short_text = "short"

    # Embed both texts
    embeddings, provider_name = await router.embed([long_text, short_text])

    # Verify embeddings were generated
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768

    # Verify provider's embed was called with multiple chunks for long text
    assert mock_provider.embed.called
    call_args = mock_provider.embed.call_args[0][0]

    # Should have multiple chunks for long text plus the short text
    assert len(call_args) > 2

    # Verify each chunk is under the token limit
    for text in call_args:
        assert estimate_tokens(text) <= 100

    # Verify the short text is included as-is
    assert short_text in call_args


@pytest.mark.asyncio
async def test_router_logs_splitting():
    """Test that router logs info when splitting long texts."""
    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.name = "test_provider"
    mock_provider.get_embedding_dimension.return_value = 768
    mock_provider.get_max_embedding_tokens.return_value = 50  # Very small limit

    async def mock_embed(texts):
        return [[0.1] * 768 for _ in texts]

    mock_provider.embed = AsyncMock(side_effect=mock_embed)

    # Create router
    router = ProviderRouter(strategy="local-first", use_cache=False)
    router.providers = [mock_provider]

    # Create text that exceeds limit
    long_text = " ".join(["word"] * 100)  # ~133 tokens

    # Embed - this should trigger splitting and logging
    embeddings, _ = await router.embed([long_text])

    # Verify embeddings were generated (splitting didn't break it)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768

    # Verify provider's embed was called with multiple chunks
    assert mock_provider.embed.called
    call_args = mock_provider.embed.call_args[0][0]

    # Should have multiple chunks, each under limit
    assert len(call_args) > 1
    for chunk in call_args:
        assert estimate_tokens(chunk) <= 50
