import pytest
from app.core.entities import BaseEntity, ChunkEntity, FileEntity

def test_base_entity_creation():
    """Test base entity creation."""
    entity = BaseEntity(
        entity_id="test-123",
        entity_type="document",
        content="Test content"
    )

    assert entity.entity_id == "test-123"
    assert entity.entity_type == "document"
    assert entity.content == "Test content"
    assert entity.created_at is not None

def test_chunk_entity_id_generation():
    """Test chunk ID generation."""
    chunk = ChunkEntity(
        parent_id="parent-123",
        chunk_index=5,
        content="Chunk content"
    )

    assert chunk.chunk_id == "parent-123_chunk_5"

def test_file_entity_defaults():
    """Test file entity type defaults."""
    file_entity = FileEntity(
        entity_id="file-1",
        entity_type="file",  # Required by BaseEntity
        content="File content",
        file_path="/path/to/file.txt"
    )

    assert file_entity.entity_type == "file"
    assert file_entity.file_path == "/path/to/file.txt"