"""Collection management endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from app.models.database import Collection, Organization
from app.models.schemas import CollectionCreate, CollectionResponse
from app.api.dependencies import get_db, get_qdrant_client
from app.storage.qdrant import QdrantStorage

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
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

    # Determine vector dimension based on provider
    if collection.vector_dimension is None:
        dimension_map = {
            "ollama": 768,
            "gemini": 768,
            "openai": 1536
        }
        collection.vector_dimension = dimension_map.get(collection.embedding_provider, 768)

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
async def list_collections(
    db: AsyncSession = Depends(get_db)
):
    """List all collections."""
    result = await db.execute(select(Collection))
    collections = result.scalars().all()
    return collections


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
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
async def delete_collection(
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
