import pytest
from app.core.entities import BaseEntity, ChunkEntity
from app.core.chunking import get_chunker
from app.core.providers.router import get_provider_router
from app.core.hashing import hash_content

@pytest.mark.asyncio
async def test_full_ingestion_pipeline():
    """Test: Entity → Chunk → Embed → Store."""

    # 1. Create entity
    entity = BaseEntity(
        entity_id="test-doc-1",
        entity_type="document",
        content="This is a test document. " * 100,  # ~200 tokens
        title="Test Document"
    )

    # 2. Compute hash
    content_hash = hash_content(entity.content)
    assert len(content_hash) == 64

    # 3. Chunk content
    chunker = get_chunker()
    chunk_texts = chunker.chunk(entity.content)
    assert len(chunk_texts) >= 1

    # 4. Embed chunks
    router = get_provider_router()
    embeddings, provider = await router.embed(chunk_texts)
    assert len(embeddings) == len(chunk_texts)

    # 5. Create chunk entities
    chunks = []
    for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
        chunk = ChunkEntity(
            parent_id=entity.entity_id,
            chunk_index=i,
            content=text,
            title=entity.title,
            embedding=embedding
        )
        chunks.append(chunk)

    assert len(chunks) == len(chunk_texts)
    print(f"✓ Processed {len(chunks)} chunks from 1 entity")