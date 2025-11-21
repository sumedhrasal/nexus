"""API dependencies for dependency injection."""

from typing import AsyncGenerator, Optional
from datetime import datetime
from uuid import UUID
from fastapi import Header, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.storage.postgres import get_db as get_db_session
from app.storage.qdrant import get_qdrant, QdrantStorage
from app.core.providers.router import get_provider_router, ProviderRouter
from app.core.auth import verify_api_key, extract_api_key_from_header
from app.models.database import APIKey, Organization, Collection
from app.core.logging import get_logger

logger = get_logger(__name__)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async for session in get_db_session():
        yield session


async def get_qdrant_client() -> QdrantStorage:
    """Get Qdrant client."""
    return get_qdrant()


async def get_embedding_router() -> ProviderRouter:
    """Get provider router."""
    return get_provider_router()


async def get_collection_router(
    collection_id: UUID,
    provider_override: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> ProviderRouter:
    """Get provider router filtered by collection's embedding_provider.

    This ensures embeddings are generated using the same provider as the collection.

    Args:
        collection_id: Collection UUID
        provider_override: Optional provider name to override collection setting
        db: Database session

    Returns:
        ProviderRouter filtered to collection's provider

    Raises:
        HTTPException: If collection not found or provider not available
    """
    # Get collection from database
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Determine which provider to use
    provider_name = provider_override or collection.embedding_provider

    # Get global router
    router = get_provider_router()

    # Filter to specific provider
    try:
        filtered_router = router.filter_by_provider_name(provider_name)
        logger.info(
            "collection_router_created",
            collection_id=str(collection_id),
            provider=provider_name,
            overridden=bool(provider_override)
        )
        return filtered_router
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


async def require_api_key(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> Organization:
    """Dependency for routes that require API key authentication.

    Usage:
        @router.get("/protected")
        async def protected_route(org: Organization = Depends(require_api_key)):
            ...

    Args:
        authorization: Authorization header (Bearer nx_xxxxx or nx_xxxxx)
        db: Database session (injected by FastAPI)

    Returns:
        Organization associated with the API key

    Raises:
        HTTPException: If API key is invalid or missing
    """
    # Extract API key from header
    api_key = extract_api_key_from_header(authorization)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide API key in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Find all active API keys and check hash
    result = await db.execute(
        select(APIKey).where(APIKey.is_active == True)
    )
    api_keys = result.scalars().all()

    matched_key = None
    for key in api_keys:
        if verify_api_key(api_key, key.key_hash):
            matched_key = key
            break

    if not matched_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last_used_at
    matched_key.last_used_at = datetime.now()
    await db.commit()

    # Get organization
    result = await db.execute(
        select(Organization).where(Organization.id == matched_key.organization_id)
    )
    organization = result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Organization not found for API key"
        )

    return organization
