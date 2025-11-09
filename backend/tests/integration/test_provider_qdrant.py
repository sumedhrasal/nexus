import pytest
from app.core.providers.router import get_provider_router
from app.storage.qdrant import get_qdrant
from app.core.entities import ChunkEntity
import uuid

@pytest.mark.asyncio
async def test_embed_and_store():
    """Test embedding generation and Qdrant storage."""
    # Get provider
    router = get_provider_router()

    # Embed texts
    texts = ["Document 1 content", "Document 2 content"]
    embeddings, provider = await router.embed(texts)

    # Create chunks
    chunks = [
        ChunkEntity(
            parent_id="parent-1",
            chunk_index=i,
            content=texts[i],
            embedding=embeddings[i]
        )
        for i in range(len(texts))
    ]

    # Store in Qdrant
    qdrant = get_qdrant()
    collection_id = str(uuid.uuid4())

    await qdrant.create_collection(
        collection_id=collection_id,
        vector_dimension=len(embeddings[0])
    )

    await qdrant.upsert_chunks(collection_id, chunks)

    # Verify storage
    count = await qdrant.count_points(collection_id)
    assert count == 2

    # Cleanup
    await qdrant.delete_collection(collection_id)