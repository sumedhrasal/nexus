"""Collection management endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.database import Collection, Organization
from app.models.schemas import CollectionCreate, CollectionResponse
from app.api.dependencies import get_db, get_qdrant_client
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import get_provider_router
from app.core.logging import get_logger

router = APIRouter(prefix="/collections", tags=["collections"])
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")  # 10 collection creates per minute
async def create_collection(
    request: Request,
    collection: CollectionCreate,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client)
):
    """Create a new collection."""

    # Get or create default organization (for now, single-tenant)
    result = await db.execute(select(Organization).limit(1))
    org = result.scalar_one_or_none()

    if not org:
        org = Organization(name="Default Organization")
        db.add(org)
        await db.commit()
        await db.refresh(org)

    # Auto-detect vector dimension from provider if not specified
    if collection.vector_dimension is None:
        try:
            # Get global provider router
            router = get_provider_router()

            # Filter to the requested provider
            filtered_router = router.filter_by_provider_name(collection.embedding_provider)

            # Get dimension from the provider
            collection.vector_dimension = filtered_router.get_primary_dimension()

            logger.info(
                "auto_detected_dimension",
                provider=collection.embedding_provider,
                dimension=collection.vector_dimension
            )
        except ValueError as e:
            # Provider not available
            logger.error(
                "provider_not_available",
                provider=collection.embedding_provider,
                error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{collection.embedding_provider}' is not available. {str(e)}"
            )
        except Exception as e:
            logger.error(
                "dimension_detection_failed",
                provider=collection.embedding_provider,
                error=str(e),
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to detect dimension for provider '{collection.embedding_provider}': {str(e)}"
            )

    # Create collection in database
    db_collection = Collection(
        organization_id=org.id,
        name=collection.name,
        embedding_provider=collection.embedding_provider,
        vector_dimension=collection.vector_dimension
    )
    db.add(db_collection)
    await db.commit()
    await db.refresh(db_collection)

    # Create collection in Qdrant
    await qdrant.create_collection(
        collection_id=str(db_collection.id),
        vector_dimension=collection.vector_dimension
    )

    return db_collection


@router.get("", response_model=List[CollectionResponse])
@limiter.limit("60/minute")  # 60 collection list requests per minute
async def list_collections(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """List all collections."""
    result = await db.execute(select(Collection))
    collections = result.scalars().all()
    return collections


@router.get("/{collection_id}", response_model=CollectionResponse)
@limiter.limit("100/minute")  # 100 collection get requests per minute
async def get_collection(
    request: Request,
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get collection by ID."""
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    return collection


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("10/minute")  # 10 collection deletes per minute
async def delete_collection(
    request: Request,
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client)
):
    """Delete a collection."""
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Delete from Qdrant
    await qdrant.delete_collection(str(collection_id))

    # Delete from database
    await db.delete(collection)
    await db.commit()
