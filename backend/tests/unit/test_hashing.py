from app.core.hashing import hash_content

def test_hash_content_deterministic():
    """Test hash is deterministic."""
    content = "Test content"

    hash1 = hash_content(content)
    hash2 = hash_content(content)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest length

def test_hash_content_different():
    """Test different content produces different hash."""
    hash1 = hash_content("Content 1")
    hash2 = hash_content("Content 2")

    assert hash1 != hash2

def test_hash_content_with_metadata():
    """Test hash includes metadata."""
    content = "Same content"
    meta1 = {"key": "value1"}
    meta2 = {"key": "value2"}

    hash1 = hash_content(content, meta1)
    hash2 = hash_content(content, meta2)

    assert hash1 != hash2  # Different metadata = different hash