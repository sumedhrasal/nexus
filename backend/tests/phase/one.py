import asyncio
from app.core.providers.ollama import OllamaProvider
from app.core.providers.router import get_provider_router
from app.core.chunking import SmartChunker
from app.storage.qdrant import QdrantStorage
import uuid

async def test_phase1():
    print("=== Testing Phase 1 Components ===\n")

    # Test 1: Provider
    print("1. Testing Ollama Provider...")
    provider = OllamaProvider()
    embeddings = await provider.embed(["Hello world", "Test text"])
    print(f"   ✓ Generated {len(embeddings)} embeddings")
    print(f"   ✓ Dimension: {len(embeddings[0])}")

    # Test 2: Provider Router
    print("\n2. Testing Provider Router...")
    router = get_provider_router()
    embeddings, provider_name = await router.embed(["Test text"])
    print(f"   ✓ Used provider: {provider_name}")
    print(f"   ✓ Embedding dimension: {len(embeddings[0])}")

    # Test 3: Chunking
    print("\n3. Testing Smart Chunker...")
    chunker = SmartChunker(max_tokens=100)
    text = "This is a test document. " * 50
    chunks = chunker.chunk(text)
    print(f"   ✓ Created {len(chunks)} chunks")
    print(f"   ✓ First chunk: {chunker.count_tokens(chunks[0])} tokens")

    # Test 4: Qdrant
    print("\n4. Testing Qdrant Storage...")
    qdrant = QdrantStorage()
    collection_id = str(uuid.uuid4())

    await qdrant.create_collection(collection_id, vector_dimension=768)
    print(f"   ✓ Created collection: {collection_id[:8]}...")

    # Create dummy chunk
    from app.core.entities import ChunkEntity
    chunk = ChunkEntity(
        parent_id="test-doc",
        chunk_index=0,
        content="Test content",
        embedding=[0.1] * 768
    )

    await qdrant.upsert_chunks(collection_id, [chunk])
    print(f"   ✓ Upserted chunk")

    count = await qdrant.count_points(collection_id)
    print(f"   ✓ Collection has {count} point(s)")

    # Search
    results = await qdrant.search(
        collection_id,
        query_vector=[0.1] * 768,
        limit=10
    )
    print(f"   ✓ Search returned {len(results)} result(s)")

    # Cleanup
    await qdrant.delete_collection(collection_id)
    print(f"   ✓ Cleaned up collection")

    print("\n=== All Phase 1 Tests Passed! ✅ ===")

# Run tests
asyncio.run(test_phase1())