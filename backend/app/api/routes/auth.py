"""Authentication and API key management endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from app.models.database import APIKey, Organization
from app.models.schemas import APIKeyCreate, APIKeyResponse, APIKeyListItem
from app.api.dependencies import get_db, require_api_key, get_embedding_router
from app.core.auth import generate_api_key
from app.core.providers.router import ProviderRouter

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreate,
    db: AsyncSession = Depends(get_db),
    current_org: Organization = Depends(require_api_key)
):
    """Create a new API key for the organization.

    **Important**: The API key is only returned once. Save it securely.

    Args:
        request: API key creation request
        db: Database session
        current_org: Current organization (from auth)

    Returns:
        API key response with the plain key
    """
    # Generate new API key
    plain_key, hashed_key = generate_api_key()

    # Create database record
    db_api_key = APIKey(
        organization_id=current_org.id,
        key_hash=hashed_key,
        name=request.name,
        is_active=True
    )
    db.add(db_api_key)
    await db.commit()
    await db.refresh(db_api_key)

    # Return response with plain key (only time it's exposed)
    return APIKeyResponse(
        id=db_api_key.id,
        organization_id=db_api_key.organization_id,
        key=plain_key,
        name=db_api_key.name,
        created_at=db_api_key.created_at
    )


@router.get("/cache/stats")
async def get_cache_stats(
    router: ProviderRouter = Depends(get_embedding_router),
    current_org: Organization = Depends(require_api_key)
):
    """Get embedding cache statistics.

    Args:
        router: Provider router instance
        current_org: Current organization (from auth)

    Returns:
        Cache statistics including hits, misses, and hit rate
    """
    stats = router.get_cache_stats()

    if stats is None:
        return {
            "cache_enabled": False,
            "message": "Embedding cache is not enabled"
        }

    return {
        "cache_enabled": True,
        **stats
    }


@router.get("/keys", response_model=List[APIKeyListItem])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    current_org: Organization = Depends(require_api_key)
):
    """List all API keys for the organization.

    Does not expose the actual key values.

    Args:
        db: Database session
        current_org: Current organization (from auth)

    Returns:
        List of API keys
    """
    result = await db.execute(
        select(APIKey).where(
            APIKey.organization_id == current_org.id
        ).order_by(APIKey.created_at.desc())
    )
    api_keys = result.scalars().all()
    return api_keys


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_org: Organization = Depends(require_api_key)
):
    """Revoke (deactivate) an API key.

    Args:
        key_id: API key ID to revoke
        db: Database session
        current_org: Current organization (from auth)

    Raises:
        HTTPException: If key not found or doesn't belong to organization
    """
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.organization_id == current_org.id
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    # Deactivate instead of deleting (keeps audit trail)
    api_key.is_active = False
    await db.commit()


@router.post("/keys/{key_id}/activate", status_code=status.HTTP_200_OK)
async def activate_api_key(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_org: Organization = Depends(require_api_key)
):
    """Reactivate a revoked API key.

    Args:
        key_id: API key ID to activate
        db: Database session
        current_org: Current organization (from auth)

    Raises:
        HTTPException: If key not found or doesn't belong to organization
    """
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.organization_id == current_org.id
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    api_key.is_active = True
    await db.commit()

    return {"status": "activated", "key_id": str(key_id)}


@router.post("/bootstrap", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def bootstrap_organization(
    db: AsyncSession = Depends(get_db)
):
    """Bootstrap a new organization with an initial API key.

    This is a special endpoint that doesn't require authentication.
    Use it to create your first organization and API key.

    **Security Note**: In production, this should be disabled or protected
    after initial setup.

    Returns:
        API key response with the plain key
    """
    # Check if any organizations exist
    result = await db.execute(select(Organization))
    existing_orgs = result.scalars().all()

    if existing_orgs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization already exists. Use /auth/keys to create additional API keys."
        )

    # Create default organization
    organization = Organization(name="Default Organization")
    db.add(organization)
    await db.commit()
    await db.refresh(organization)

    # Generate API key
    plain_key, hashed_key = generate_api_key()

    # Create API key
    db_api_key = APIKey(
        organization_id=organization.id,
        key_hash=hashed_key,
        name="Bootstrap Key",
        is_active=True
    )
    db.add(db_api_key)
    await db.commit()
    await db.refresh(db_api_key)

    return APIKeyResponse(
        id=db_api_key.id,
        organization_id=db_api_key.organization_id,
        key=plain_key,
        name=db_api_key.name,
        created_at=db_api_key.created_at
    )
