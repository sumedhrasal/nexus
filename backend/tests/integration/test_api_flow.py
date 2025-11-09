"""Critical path integration test for API flow."""

import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_complete_flow():
    """Test: Create collection → Ingest documents → Search → Verify results."""

    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. Create collection
        create_response = await client.post(
            "/collections",
            json={
                "name": "Test Collection",
                "embedding_provider": "ollama",
                "vector_dimension": 768
            }
        )
        assert create_response.status_code == 201
        collection_data = create_response.json()
        collection_id = collection_data["id"]
        assert collection_data["name"] == "Test Collection"
        assert collection_data["embedding_provider"] == "ollama"

        # 2. Ingest documents
        ingest_response = await client.post(
            f"/collections/{collection_id}/ingest",
            json={
                "documents": [
                    {
                        "content": "Python is a high-level programming language known for its simplicity and readability.",
                        "title": "Python Overview",
                        "metadata": {"category": "programming"}
                    },
                    {
                        "content": "JavaScript is a versatile language used for web development and server-side programming.",
                        "title": "JavaScript Overview",
                        "metadata": {"category": "programming"}
                    },
                    {
                        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                        "title": "ML Basics",
                        "metadata": {"category": "ai"}
                    }
                ]
            }
        )
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert ingest_data["documents_processed"] == 3
        assert ingest_data["chunks_created"] >= 3
        assert ingest_data["entities_inserted"] == 3

        # 3. Search for Python content
        search_response = await client.post(
            f"/collections/{collection_id}/search",
            json={
                "query": "Tell me about Python programming language",
                "limit": 5,
                "use_cache": True
            }
        )
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert search_data["total_results"] > 0
        assert search_data["from_cache"] is False  # First search

        # Verify result relevance
        top_result = search_data["results"][0]
        assert "Python" in top_result["content"]
        assert top_result["score"] > 0

        # 4. Search again (should hit cache)
        search_response_2 = await client.post(
            f"/collections/{collection_id}/search",
            json={
                "query": "Tell me about Python programming language",
                "limit": 5,
                "use_cache": True
            }
        )
        assert search_response_2.status_code == 200
        search_data_2 = search_response_2.json()
        assert search_data_2["from_cache"] is True  # Cached
        assert search_data_2["results"] == search_data["results"]  # Same results

        # 5. List collections
        list_response = await client.get("/collections")
        assert list_response.status_code == 200
        collections = list_response.json()
        assert len(collections) >= 1
        assert any(c["id"] == collection_id for c in collections)

        # 6. Get specific collection
        get_response = await client.get(f"/collections/{collection_id}")
        assert get_response.status_code == 200
        get_data = get_response.json()
        assert get_data["id"] == collection_id
        assert get_data["name"] == "Test Collection"

        # 7. Delete collection
        delete_response = await client.delete(f"/collections/{collection_id}")
        assert delete_response.status_code == 204

        # 8. Verify deletion
        get_deleted = await client.get(f"/collections/{collection_id}")
        assert get_deleted.status_code == 404
