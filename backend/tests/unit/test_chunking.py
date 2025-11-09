"""Unit tests for chunking."""

import pytest
from app.core.chunking import SemanticChunker, SmartChunker


def test_semantic_chunker_basic():
    """Test basic chunking."""
    chunker = SemanticChunker(max_tokens=100, overlap=10)

    text = "This is a test. " * 50  # ~100 tokens
    chunks = chunker.chunk(text)

    assert len(chunks) >= 1
    assert all(chunker.count_tokens(chunk) <= 100 for chunk in chunks)


def test_semantic_chunker_short_text():
    """Test chunking short text that fits in one chunk."""
    chunker = SemanticChunker(max_tokens=100, overlap=10)

    text = "Short text."
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_semantic_chunker_overlap():
    """Test chunk overlap."""
    chunker = SemanticChunker(max_tokens=50, overlap=10)

    text = "Word " * 100  # Long text
    chunks = chunker.chunk(text)

    # Should create multiple chunks
    assert len(chunks) > 1


def test_smart_chunker_paragraphs():
    """Test smart chunker preserves paragraphs."""
    chunker = SmartChunker(max_tokens=100, overlap=10)

    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = chunker.chunk(text)

    # Should preserve some paragraph boundaries
    assert len(chunks) >= 1


def test_chunker_empty_input():
    """Test chunker handles empty input."""
    chunker = SemanticChunker()

    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunker_token_counting():
    """Test token counting."""
    chunker = SemanticChunker()

    text = "Hello world"
    token_count = chunker.count_tokens(text)

    assert token_count > 0
    assert isinstance(token_count, int)
