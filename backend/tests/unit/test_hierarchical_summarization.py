"""Unit tests for hierarchical summarization with mocked LLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List

from app.compression.hierarchical_summarization import (
    HierarchicalSummarizer,
    SummaryLayer
)


@pytest.fixture
def mock_provider():
    """Create a mock provider router."""
    provider = MagicMock()

    # Mock generate method for summaries
    async def mock_generate(prompt: str, system: str = ""):
        # Return simple summary based on prompt length
        if "section" in prompt.lower():
            return "Section summary: Key concepts and main ideas.", {}
        else:
            return "Document summary: High-level overview of main themes.", {}

    # Mock embed method for embeddings
    async def mock_embed(texts: List[str], required_dimension: int = 1024):
        # Return fake embeddings with correct dimension
        embeddings = [[0.1] * required_dimension for _ in texts]
        return embeddings, "mock-provider"

    provider.generate = AsyncMock(side_effect=mock_generate)
    provider.embed = AsyncMock(side_effect=mock_embed)

    return provider


@pytest.mark.asyncio
async def test_build_hierarchy_structure(mock_provider):
    """Test that hierarchy builds correct 3-layer structure."""
    summarizer = HierarchicalSummarizer(provider=mock_provider, vector_dimension=1024)

    # Create test chunks (11 chunks -> should create ~2-3 sections)
    chunks = [f"This is test chunk {i} with some content." for i in range(11)]
    chunk_embeddings = [[0.1 * i] * 1024 for i in range(11)]

    metadata = {"title": "Test Document", "source": "test"}
    entity_id = "test_entity_123"

    # Build hierarchy
    all_nodes = await summarizer.build_hierarchy(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        document_metadata=metadata,
        entity_id=entity_id
    )

    # Verify structure
    chunk_nodes = [n for n in all_nodes if n.layer == SummaryLayer.CHUNK]
    section_nodes = [n for n in all_nodes if n.layer == SummaryLayer.SECTION]
    document_nodes = [n for n in all_nodes if n.layer == SummaryLayer.DOCUMENT]

    # Should have all 11 chunk nodes
    assert len(chunk_nodes) == 11, f"Expected 11 chunk nodes, got {len(chunk_nodes)}"

    # Should have 2-3 section nodes (11 chunks / ~5 chunks per section)
    assert 2 <= len(section_nodes) <= 3, f"Expected 2-3 section nodes, got {len(section_nodes)}"

    # Should have exactly 1 document node
    assert len(document_nodes) == 1, f"Expected 1 document node, got {len(document_nodes)}"


@pytest.mark.asyncio
async def test_hierarchy_parent_child_links(mock_provider):
    """Test that parent-child links are correctly established."""
    summarizer = HierarchicalSummarizer(provider=mock_provider, vector_dimension=1024)

    chunks = [f"Chunk {i}" for i in range(6)]
    chunk_embeddings = [[0.1 * i] * 1024 for i in range(6)]
    metadata = {"title": "Test"}

    all_nodes = await summarizer.build_hierarchy(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        document_metadata=metadata,
        entity_id="test_123"
    )

    chunk_nodes = [n for n in all_nodes if n.layer == SummaryLayer.CHUNK]
    section_nodes = [n for n in all_nodes if n.layer == SummaryLayer.SECTION]
    document_node = [n for n in all_nodes if n.layer == SummaryLayer.DOCUMENT][0]

    # All chunks should have parent_id pointing to a section
    for chunk in chunk_nodes:
        assert chunk.parent_id is not None
        assert chunk.parent_id.startswith("test_123_section_")

    # All sections should have parent_id pointing to document
    for section in section_nodes:
        assert section.parent_id == "test_123_document"
        assert len(section.children_ids) > 0  # Should have children chunks

    # Document should have no parent
    assert document_node.parent_id is None
    assert len(document_node.children_ids) == len(section_nodes)


@pytest.mark.asyncio
async def test_hierarchy_node_ids(mock_provider):
    """Test that node IDs follow expected naming convention."""
    summarizer = HierarchicalSummarizer(provider=mock_provider, vector_dimension=1024)

    chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
    chunk_embeddings = [[0.1, 0.2] * 512 for _ in range(3)]
    metadata = {"title": "Test"}
    entity_id = "entity_abc"

    all_nodes = await summarizer.build_hierarchy(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        document_metadata=metadata,
        entity_id=entity_id
    )

    # Verify node ID patterns
    chunk_nodes = [n for n in all_nodes if n.layer == SummaryLayer.CHUNK]
    section_nodes = [n for n in all_nodes if n.layer == SummaryLayer.SECTION]
    document_node = [n for n in all_nodes if n.layer == SummaryLayer.DOCUMENT][0]

    # Chunk IDs: entity_abc_chunk_0, entity_abc_chunk_1, etc.
    for i, chunk in enumerate(chunk_nodes):
        assert chunk.node_id == f"{entity_id}_chunk_{i}"
        assert chunk.original_chunk_index == i

    # Section IDs: entity_abc_section_0, entity_abc_section_1, etc.
    for section in section_nodes:
        assert section.node_id.startswith(f"{entity_id}_section_")
        assert section.original_chunk_index is None  # Sections don't have this

    # Document ID: entity_abc_document
    assert document_node.node_id == f"{entity_id}_document"


@pytest.mark.asyncio
async def test_hierarchy_embeddings_generated(mock_provider):
    """Test that embeddings are generated for all nodes."""
    summarizer = HierarchicalSummarizer(provider=mock_provider, vector_dimension=1024)

    chunks = ["Chunk 1", "Chunk 2"]
    chunk_embeddings = [[0.5] * 1024, [0.6] * 1024]
    metadata = {"title": "Test"}

    all_nodes = await summarizer.build_hierarchy(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        document_metadata=metadata,
        entity_id="test"
    )

    # All nodes should have embeddings with correct dimension
    for node in all_nodes:
        assert len(node.embedding) == 1024
        assert all(isinstance(x, float) for x in node.embedding)


@pytest.mark.asyncio
async def test_hierarchy_with_few_chunks(mock_provider):
    """Test hierarchy building with only 2-3 chunks (edge case)."""
    summarizer = HierarchicalSummarizer(provider=mock_provider, vector_dimension=1024)

    chunks = ["Chunk 1", "Chunk 2"]
    chunk_embeddings = [[0.1] * 1024, [0.2] * 1024]
    metadata = {"title": "Small Doc"}

    all_nodes = await summarizer.build_hierarchy(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        document_metadata=metadata,
        entity_id="small"
    )

    # Should still create all 3 layers
    chunk_nodes = [n for n in all_nodes if n.layer == SummaryLayer.CHUNK]
    section_nodes = [n for n in all_nodes if n.layer == SummaryLayer.SECTION]
    document_nodes = [n for n in all_nodes if n.layer == SummaryLayer.DOCUMENT]

    assert len(chunk_nodes) == 2
    assert len(section_nodes) == 1  # Should create single section for few chunks
    assert len(document_nodes) == 1


@pytest.mark.asyncio
async def test_llm_generate_called_correctly(mock_provider):
    """Test that LLM generate is called with correct prompts."""
    summarizer = HierarchicalSummarizer(provider=mock_provider, vector_dimension=512)

    chunks = ["Test chunk 1", "Test chunk 2", "Test chunk 3"]
    chunk_embeddings = [[0.1] * 512 for _ in range(3)]
    metadata = {"title": "Test Title"}

    await summarizer.build_hierarchy(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        document_metadata=metadata,
        entity_id="test"
    )

    # Verify generate was called (at least once for sections and once for document)
    assert mock_provider.generate.call_count >= 2

    # Verify embed was called for sections and document (not for chunks - they're provided)
    assert mock_provider.embed.call_count >= 2


@pytest.mark.asyncio
async def test_metadata_propagated_to_nodes(mock_provider):
    """Test that metadata is propagated to all nodes."""
    summarizer = HierarchicalSummarizer(provider=mock_provider, vector_dimension=1024)

    chunks = ["Chunk 1"]
    chunk_embeddings = [[0.1] * 1024]
    metadata = {"title": "Important Doc", "author": "Test Author", "date": "2024-01-01"}

    all_nodes = await summarizer.build_hierarchy(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        document_metadata=metadata,
        entity_id="test"
    )

    # All nodes should have the metadata
    for node in all_nodes:
        assert node.metadata["title"] == "Important Doc"
        assert node.metadata["author"] == "Test Author"
        assert node.metadata["date"] == "2024-01-01"
