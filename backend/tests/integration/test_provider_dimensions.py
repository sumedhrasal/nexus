"""Test provider dimension detection.

This test validates that each provider correctly reports its embedding dimension:
- Gemini: models/gemini-embedding-001 -> 3072 dimensions
- OpenAI: text-embedding-3-small -> 1536 dimensions
- Ollama: configurable via settings (currently 1024 dimensions for nomic-embed-text:latest)
"""

import pytest
from app.core.providers.router import get_provider_router
from app.config import settings


@pytest.mark.asyncio
async def test_gemini_provider_dimension():
    """Test that Gemini provider reports correct dimension."""
    if not settings.gemini_api_key:
        pytest.skip("Gemini API key not configured")

    router = get_provider_router()

    # Filter to Gemini provider
    try:
        gemini_router = router.filter_by_provider_name("gemini")
        dimension = gemini_router.get_primary_dimension()

        assert dimension == settings.gemini_embedding_dimension, (
            f"Expected Gemini dimension {settings.gemini_embedding_dimension}, "
            f"got {dimension}"
        )

        print(f"\n✅ Gemini provider dimension: {dimension}")
        print(f"   Model: {settings.gemini_embedding_model}")
        print(f"   Expected: {settings.gemini_embedding_dimension}")
    except ValueError as e:
        pytest.skip(f"Gemini provider not available: {e}")


@pytest.mark.asyncio
async def test_openai_provider_dimension():
    """Test that OpenAI provider reports correct dimension."""
    if not settings.openai_api_key:
        pytest.skip("OpenAI API key not configured")

    router = get_provider_router()

    # Filter to OpenAI provider
    try:
        openai_router = router.filter_by_provider_name("openai")
        dimension = openai_router.get_primary_dimension()

        assert dimension == settings.openai_embedding_dimension, (
            f"Expected OpenAI dimension {settings.openai_embedding_dimension}, "
            f"got {dimension}"
        )

        print(f"\n✅ OpenAI provider dimension: {dimension}")
        print(f"   Model: {settings.openai_embedding_model}")
        print(f"   Expected: {settings.openai_embedding_dimension}")
    except ValueError as e:
        pytest.skip(f"OpenAI provider not available: {e}")


@pytest.mark.asyncio
async def test_ollama_provider_dimension():
    """Test that Ollama provider reports correct dimension."""
    router = get_provider_router()

    # Filter to Ollama provider
    try:
        ollama_router = router.filter_by_provider_name("ollama")
        dimension = ollama_router.get_primary_dimension()

        # Ollama dimension is configurable
        expected = settings.ollama_embedding_dimension
        assert dimension == expected, (
            f"Expected Ollama dimension {expected}, got {dimension}"
        )

        print(f"\n✅ Ollama provider dimension: {dimension}")
        print(f"   Model: {settings.ollama_embedding_model}")
        print(f"   Expected: {expected}")
    except ValueError as e:
        pytest.skip(f"Ollama provider not available: {e}")


@pytest.mark.asyncio
async def test_provider_info():
    """Test get_provider_info returns correct dimensions for all providers."""
    router = get_provider_router()
    info = router.get_provider_info()

    print(f"\n📊 Provider Info:")
    for provider_info in info:
        print(f"   • {provider_info['name']} ({provider_info['type']})")
        print(f"     Dimension: {provider_info['dimension']}")
        print(f"     Cost/1K: ${provider_info['cost_per_1k']}")

    # Verify all providers have valid dimensions
    for provider_info in info:
        assert provider_info['dimension'] > 0, (
            f"Provider {provider_info['name']} has invalid dimension: {provider_info['dimension']}"
        )
        assert isinstance(provider_info['dimension'], int), (
            f"Provider {provider_info['name']} dimension is not an integer"
        )
