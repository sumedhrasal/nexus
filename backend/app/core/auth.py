"""Authentication utilities for API key management."""

import secrets
from typing import Optional, Tuple
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_api_key() -> Tuple[str, str]:
    """Generate a new API key and its hash.

    Returns:
        Tuple of (plain_key, hashed_key)

    Example:
        plain_key: "nx_1a2b3c4d5e6f7g8h9i0j..."
        hashed_key: "$2b$12$..."
    """
    # Generate random key (32 bytes = 43 chars in base64)
    random_part = secrets.token_urlsafe(32)

    # Add prefix for identification
    plain_key = f"nx_{random_part}"

    # Hash the key for storage
    hashed_key = hash_api_key(plain_key)

    return plain_key, hashed_key


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage.

    Args:
        api_key: Plain text API key

    Returns:
        Bcrypt hash of the API key

    Note:
        Bcrypt has a 72-byte limit. Passlib with bcrypt 4.x handles truncation automatically.
    """
    return pwd_context.hash(api_key)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash.

    Args:
        plain_key: Plain text API key from request
        hashed_key: Stored hash from database

    Returns:
        True if key matches, False otherwise

    Note:
        Bcrypt 4.x with passlib handles truncation automatically.
    """
    return pwd_context.verify(plain_key, hashed_key)


def extract_api_key_from_header(authorization: Optional[str]) -> Optional[str]:
    """Extract API key from Authorization header.

    Supports both formats:
    - Bearer nx_xxxxx
    - nx_xxxxx

    Args:
        authorization: Authorization header value

    Returns:
        API key if valid format, None otherwise
    """
    if not authorization:
        return None

    # Remove "Bearer " prefix if present
    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
    else:
        api_key = authorization

    # Validate format (should start with nx_)
    if not api_key.startswith("nx_"):
        return None

    return api_key
