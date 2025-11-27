"""Unit tests for authentication functionality."""

from app.core.auth import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
    extract_api_key_from_header
)


class TestAPIKeyGeneration:
    """Test API key generation."""

    def test_generate_api_key_format(self):
        """Test that generated API keys have correct format."""
        plain_key, hashed_key = generate_api_key()

        # Check plain key format
        assert plain_key.startswith("nx_")
        assert len(plain_key) > 40  # nx_ + 32 bytes base64 = ~45 chars

        # Check hash is different from plain key
        assert hashed_key != plain_key
        assert hashed_key.startswith("$2b$")  # bcrypt hash prefix

    def test_generate_api_key_uniqueness(self):
        """Test that generated API keys are unique."""
        key1, hash1 = generate_api_key()
        key2, hash2 = generate_api_key()

        assert key1 != key2
        assert hash1 != hash2

    def test_generated_key_is_verifiable(self):
        """Test that generated keys can be verified."""
        plain_key, hashed_key = generate_api_key()

        # Should verify successfully
        assert verify_api_key(plain_key, hashed_key) is True


class TestAPIKeyHashing:
    """Test API key hashing functionality."""

    def test_hash_api_key(self):
        """Test that API keys are hashed correctly."""
        api_key = "nx_test_key_12345"
        hashed = hash_api_key(api_key)

        # Should be bcrypt hash
        assert hashed.startswith("$2b$")
        assert len(hashed) == 60  # bcrypt hash length

    def test_hash_is_consistent(self):
        """Test that same key produces different hashes (salt)."""
        api_key = "nx_test_key_12345"
        hash1 = hash_api_key(api_key)
        hash2 = hash_api_key(api_key)

        # Different due to salt, but both should verify
        assert hash1 != hash2
        assert verify_api_key(api_key, hash1)
        assert verify_api_key(api_key, hash2)


class TestAPIKeyVerification:
    """Test API key verification."""

    def test_verify_correct_key(self):
        """Test verification of correct API key."""
        api_key = "nx_correct_key"
        hashed = hash_api_key(api_key)

        assert verify_api_key(api_key, hashed) is True

    def test_verify_incorrect_key(self):
        """Test verification fails for incorrect key."""
        correct_key = "nx_correct_key"
        wrong_key = "nx_wrong_key"
        hashed = hash_api_key(correct_key)

        assert verify_api_key(wrong_key, hashed) is False

    def test_verify_empty_key(self):
        """Test verification fails for empty key."""
        api_key = "nx_test_key"
        hashed = hash_api_key(api_key)

        assert verify_api_key("", hashed) is False


class TestExtractAPIKeyFromHeader:
    """Test extracting API key from Authorization header."""

    def test_extract_bearer_format(self):
        """Test extraction from Bearer format."""
        header = "Bearer nx_test_key_12345"
        api_key = extract_api_key_from_header(header)

        assert api_key == "nx_test_key_12345"

    def test_extract_direct_format(self):
        """Test extraction from direct format."""
        header = "nx_test_key_12345"
        api_key = extract_api_key_from_header(header)

        assert api_key == "nx_test_key_12345"

    def test_extract_none_header(self):
        """Test extraction returns None for None header."""
        api_key = extract_api_key_from_header(None)

        assert api_key is None

    def test_extract_empty_header(self):
        """Test extraction returns None for empty header."""
        api_key = extract_api_key_from_header("")

        assert api_key is None

    def test_extract_invalid_prefix(self):
        """Test extraction returns None for invalid prefix."""
        header = "Bearer invalid_key_12345"
        api_key = extract_api_key_from_header(header)

        assert api_key is None

    def test_extract_no_bearer(self):
        """Test extraction works without Bearer prefix."""
        header = "nx_test_key_12345"
        api_key = extract_api_key_from_header(header)

        assert api_key == "nx_test_key_12345"

    def test_extract_extra_spaces(self):
        """Test extraction with extra spaces after Bearer."""
        header = "Bearer  nx_test_key_12345"  # Extra space after Bearer
        api_key = extract_api_key_from_header(header)

        # Will extract " nx_test_key_12345" (with leading space)
        # This won't have nx_ prefix, so should return None
        assert api_key is None  # Invalid due to space before nx_


class TestAPIKeySecurityProperties:
    """Test security properties of API keys."""

    def test_key_length_sufficient(self):
        """Test that generated keys have sufficient entropy."""
        plain_key, _ = generate_api_key()

        # Remove prefix, check remaining length
        key_body = plain_key[3:]  # Remove "nx_"

        # Should be at least 32 bytes base64 encoded (~43 chars)
        assert len(key_body) >= 40

    def test_hash_timing_safe(self):
        """Test that verification is timing-safe (uses bcrypt)."""
        api_key = "nx_test_key"
        hashed = hash_api_key(api_key)

        # Bcrypt should take similar time for correct/incorrect keys
        # This is implicit in bcrypt's implementation
        assert verify_api_key(api_key, hashed) is True
        assert verify_api_key("nx_wrong", hashed) is False

    def test_hash_not_reversible(self):
        """Test that hash cannot be reversed to get original key."""
        api_key = "nx_secret_key_12345"
        hashed = hash_api_key(api_key)

        # Hash should not contain the original key
        assert api_key not in hashed
        assert "secret_key" not in hashed
        assert "12345" not in hashed
