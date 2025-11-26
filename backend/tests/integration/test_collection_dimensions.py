"""Test collection creation with correct vector dimensions for different providers.

This test validates that the create_collection endpoint correctly auto-detects
and sets the vector dimension based on the specified embedding provider.

Note: This test focuses on API-level validation using the Gemini provider.
For comprehensive provider dimension testing across all providers (Gemini, OpenAI, Ollama),
see test_provider_dimensions.py which tests at the provider level without HTTP client issues.
"""

import pytest
from httpx import AsyncClient
from app.main import app
from app.config import settings


@pytest.mark.asyncio
async def test_gemini_collection_dimension():
    """Test that Gemini collection gets the correct dimension (3072 for gemini-embedding-001)."""
    async with AsyncClient(app=app, base_url="http://localhost:8000", timeout=30.0) as client:
        # Create collection with Gemini provider
        collection_name = "Test Gemini Dimension"

        response = await client.post(
            "/collections",
            json={
                "name": collection_name,
                "embedding_provider": "gemini",
                # Don't specify vector_dimension - let it auto-detect
            }
        )

        # Skip if Gemini not available
        if response.status_code == 400 and "not available" in response.text:
            pytest.skip("Gemini provider not available - skipping test")

        # Validate response
        assert response.status_code == 201, f"Failed to create collection: {response.text}"
        data = response.json()

        # Verify provider and dimension
        assert data["embedding_provider"] == "gemini"
        assert data["vector_dimension"] == settings.gemini_embedding_dimension, (
            f"Expected Gemini dimension {settings.gemini_embedding_dimension}, "
            f"got {data['vector_dimension']}"
        )

        print(f"\n✅ Gemini collection created: {data['id']}")
        print(f"   Provider: {data['embedding_provider']}")
        print(f"   Dimension: {data['vector_dimension']} (expected: {settings.gemini_embedding_dimension})")

        # Cleanup
        collection_id = data["id"]
        delete_response = await client.delete(f"/collections/{collection_id}")
        assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_invalid_provider_fails():
    """Test that creating collection with invalid provider fails gracefully."""
    async with AsyncClient(app=app, base_url="http://localhost:8000", timeout=30.0) as client:
        # Try to create collection with non-existent provider
        collection_name = "Test Invalid Provider"

        response = await client.post(
            "/collections",
            json={
                "name": collection_name,
                "embedding_provider": "nonexistent-provider",
            }
        )

        # Should fail with 422 (validation error) or 400 (provider not available)
        assert response.status_code in [400, 422], (
            f"Expected 400 or 422 for invalid provider, got {response.status_code}"
        )

        print(f"\n✅ Invalid provider correctly rejected")
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print(f"   Validation error (422 - schema validation failed)")
        else:
            print(f"   Message: {response.json()['detail']}")
