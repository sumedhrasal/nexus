"""Test end-to-end dimension detection and document ingestion flow.

This test validates the complete workflow:
1. Create collection with only provider name (no explicit dimension)
2. System auto-detects correct dimension from provider
3. Ingest document content
4. Verify embeddings are generated with correct dimension
5. Verify data is successfully stored in Qdrant
6. Verify data can be retrieved from Qdrant

This catches dimension mismatch issues between collection creation and ingestion.
"""

import pytest
from httpx import AsyncClient
from app.main import app
from app.config import settings


@pytest.mark.asyncio
async def test_gemini_end_to_end_ingestion():
    """Test complete Gemini ingestion flow with auto-detected dimensions."""

    async with AsyncClient(app=app, base_url="http://localhost:8000", timeout=120.0) as client:
        collection_id = None

        try:
            # Step 1: Create collection (auto-detect dimension)
            print("\n📦 Creating collection with Gemini (auto-detect dimension)...")

            create_response = await client.post(
                "/collections",
                json={
                    "name": "Gemini E2E Ingestion Test",
                    "embedding_provider": "gemini",
                }
            )

            if create_response.status_code == 400 and "not available" in create_response.text:
                pytest.skip("Gemini provider not available")

            assert create_response.status_code == 201, f"Failed to create collection: {create_response.text}"
            collection_data = create_response.json()
            collection_id = collection_data["id"]

            print(f"   ✅ Collection: {collection_id}")
            print(f"   📊 Dimension: {collection_data['vector_dimension']}")

            assert collection_data["vector_dimension"] == settings.gemini_embedding_dimension

            # Step 2: Ingest document
            print(f"\n📄 Ingesting document...")

            ingest_response = await client.post(
                f"/collections/{collection_id}/ingest",
                json={
                    "documents": [{
                        "title": "Test Doc",
                        "content": "RAG combines retrieval and generation for better AI responses."
                    }]
                }
            )

            assert ingest_response.status_code == 200, f"Ingestion failed: {ingest_response.text}"
            ingest_data = ingest_response.json()

            print(f"   ✅ Chunks: {ingest_data.get('chunks_created', 0)}")
            assert ingest_data.get("chunks_created", 0) > 0

            # Step 3: Search to verify
            print(f"\n🔍 Searching...")

            search_response = await client.post(
                f"/collections/{collection_id}/search",
                json={"query": "What is RAG?", "limit": 5}
            )

            assert search_response.status_code == 200, f"Search failed: {search_response.text}"
            search_data = search_response.json()

            print(f"   ✅ Results: {search_data['total_results']}")
            assert search_data["total_results"] > 0

            print(f"\n✅ END-TO-END TEST PASSED")

        finally:
            # Cleanup
            if collection_id:
                await client.delete(f"/collections/{collection_id}")


# @pytest.mark.asyncio
# async def test_ollama_end_to_end_ingestion():
#     """Test complete Ollama ingestion flow with auto-detected dimensions."""

#     async with AsyncClient(app=app, base_url="http://localhost:8000", timeout=120.0) as client:
#         collection_id = None

#         try:
#             # Step 1: Create collection (auto-detect dimension)
#             print("\n📦 Creating collection with Ollama (auto-detect dimension)...")

#             create_response = await client.post(
#                 "/collections",
#                 json={
#                     "name": "Ollama E2E Ingestion Test",
#                     "embedding_provider": "ollama",
#                 }
#             )

#             if create_response.status_code == 400 and "not available" in create_response.text:
#                 pytest.skip("Ollama provider not available")

#             assert create_response.status_code == 201, f"Failed to create collection: {create_response.text}"
#             collection_data = create_response.json()
#             collection_id = collection_data["id"]

#             print(f"   ✅ Collection: {collection_id}")
#             print(f"   📊 Dimension: {collection_data['vector_dimension']}")

#             assert collection_data["vector_dimension"] == settings.ollama_embedding_dimension

#             # Step 2: Ingest document
#             print(f"\n📄 Ingesting document...")

#             ingest_response = await client.post(
#                 f"/collections/{collection_id}/ingest",
#                 json={
#                     "documents": [{
#                         "title": "Test Doc",
#                         "content": "Machine learning enables computers to learn from data."
#                     }]
#                 }
#             )

#             assert ingest_response.status_code == 200, f"Ingestion failed: {ingest_response.text}"
#             ingest_data = ingest_response.json()

#             print(f"   ✅ Chunks: {ingest_data.get('chunks_created', 0)}")
#             assert ingest_data.get("chunks_created", 0) > 0

#             # Step 3: Search to verify
#             print(f"\n🔍 Searching...")

#             search_response = await client.post(
#                 f"/collections/{collection_id}/search",
#                 json={"query": "What is machine learning?", "limit": 5}
#             )

#             assert search_response.status_code == 200, f"Search failed: {search_response.text}"
#             search_data = search_response.json()

#             print(f"   ✅ Results: {search_data['total_results']}")
#             assert search_data["total_results"] > 0

#             print(f"\n✅ END-TO-END TEST PASSED")

#         finally:
#             # Cleanup
#             if collection_id:
#                 await client.delete(f"/collections/{collection_id}")
