"""API authentication middleware."""

import hashlib
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.database import APIKey
from app.config import settings
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


def hash_api_key(key: str) -> str:
    """Hash an API key for storage.

    Args:
        key: Raw API key

    Returns:
        Hashed API key
    """
    salted = f"{key}{settings.api_key_salt}"
    return hashlib.sha256(salted.encode()).hexdigest()


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    db: AsyncSession = None
) -> Optional[APIKey]:
    """Verify API key from request headers.

    Args:
        credentials: HTTP authorization credentials
        db: Database session

    Returns:
        APIKey object if valid, None otherwise

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        # For now, allow unauthenticated access for development
        # In production, uncomment this:
        # raise HTTPException(
        #     status_code=status.HTTP_401_UNAUTHORIZED,
        #     detail="Missing API key"
        # )
        logger.warning("Request without API key (development mode)")
        return None

    # Hash the provided key
    key_hash = hash_api_key(credentials.credentials)

    # Look up in database
    if db:
        result = await db.execute(
            select(APIKey).where(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            )
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        # Update last_used_at
        from datetime import datetime
        api_key.last_used_at = datetime.now()
        await db.commit()

        return api_key

    return None


def generate_api_key() -> str:
    """Generate a new API key.

    Returns:
        Random API key string
    """
    import secrets
    return f"nx_{secrets.token_urlsafe(32)}"
