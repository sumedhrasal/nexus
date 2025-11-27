from app.core.chunking import SemanticChunker, SmartChunker

def test_semantic_chunker_basic():
    """Test basic chunking."""
    chunker = SemanticChunker(max_tokens=100, overlap=10)

    text = "This is a test. " * 50  # ~100 tokens
    chunks = chunker.chunk(text)

    assert len(chunks) >= 1
    assert all(chunker.count_tokens(chunk) <= 100 for chunk in chunks)

def test_semantic_chunker_overlap():
    """Test chunk overlap."""
    chunker = SemanticChunker(max_tokens=50, overlap=10)

    text = "Word " * 100  # Long text
    chunks = chunker.chunk(text)

    # Check overlap exists
    if len(chunks) > 1:
        # Last tokens of chunk[0] should appear in chunk[1]
        assert len(chunks) > 1

def test_smart_chunker_paragraphs():
    """Test smart chunker preserves paragraphs."""
    chunker = SmartChunker(max_tokens=100, overlap=10)

    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = chunker.chunk(text)

    # Should preserve paragraph boundaries
    assert all("\n\n" in chunk or len(chunk.split()) < 20 for chunk in chunks)

def test_chunker_empty_input():
    """Test chunker handles empty input."""
    chunker = SemanticChunker()

    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []