import pytest
from app.core.providers.ollama import OllamaProvider
from app.core.providers.router import ProviderRouter
from app.config import settings

@pytest.mark.asyncio
async def test_ollama_provider_embed():
    """Test Ollama embedding generation."""
    provider = OllamaProvider()

    texts = ["Hello world", "This is a test"]
    embeddings = await provider.embed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == settings.ollama_embedding_dimension
    assert all(isinstance(x, float) for x in embeddings[0])

@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires Ollama running on host machine")
async def test_ollama_provider_generate():
    """Test Ollama text generation."""
    provider = OllamaProvider()

    prompt = "What is 2+2?"
    response = await provider.generate(prompt)

    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_provider_router_fallback():
    """Test provider router fallback mechanism."""
    router = ProviderRouter(strategy="local-first")

    # Should use Ollama first (local-first strategy)
    embeddings, provider_name = await router.embed(["test text"])

    assert provider_name == "ollama"
    assert len(embeddings) == 1
    assert len(embeddings[0]) == settings.ollama_embedding_dimension

@pytest.mark.asyncio
async def test_provider_dimension_filtering():
    """Test dimension-based provider filtering."""
    router = ProviderRouter()

    # Request specific dimension (Ollama's current dimension)
    embeddings, provider_name = await router.embed(
        ["test"],
        required_dimension=settings.ollama_embedding_dimension
    )

    assert len(embeddings[0]) == settings.ollama_embedding_dimension