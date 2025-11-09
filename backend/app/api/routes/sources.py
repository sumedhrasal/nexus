"""Source connector endpoints."""

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from pydantic import BaseModel

from app.models.database import Collection, Entity
from app.models.schemas import IngestResponse
from app.api.dependencies import get_db, get_qdrant_client, get_embedding_router
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.core.chunking import get_chunker
from app.core.hashing import hash_content
from app.core.entities import ChunkEntity
from app.core.sources.local_files import LocalFileSource
from app.core.sources.github import GitHubSource

router = APIRouter(prefix="/collections/{collection_id}/sources", tags=["sources"])


class LocalFileIngestRequest(BaseModel):
    """Request to ingest from local files."""
    path: str
    recursive: bool = True
    extensions: List[str] = None


class GitHubIngestRequest(BaseModel):
    """Request to ingest from GitHub."""
    repo_name: str
    branch: str = "main"
    path: str = ""
    extensions: List[str] = None


@router.post("/local-files", response_model=IngestResponse)
async def ingest_local_files(
    collection_id: uuid.UUID,
    request: LocalFileIngestRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client),
    router: ProviderRouter = Depends(get_embedding_router)
):
    """Ingest documents from local file system."""
    start_time = time.time()

    # Verify collection exists
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Initialize source
    source = LocalFileSource()
    chunker = get_chunker()

    # Track stats
    documents_processed = 0
    chunks_created = 0
    entities_inserted = 0
    entities_updated = 0
    all_chunk_entities: List[ChunkEntity] = []

    # Fetch and process files
    async for file_entity in source.fetch(
        path=request.path,
        recursive=request.recursive,
        extensions=request.extensions
    ):
        # Compute content hash
        content_hash = hash_content(file_entity.content)

        # Check if entity exists
        result = await db.execute(
            select(Entity).where(
                Entity.collection_id == collection_id,
                Entity.content_hash == content_hash
            )
        )
        existing_entity = result.scalar_one_or_none()

        if existing_entity:
            entities_updated += 1
            continue

        # Create entity
        db_entity = Entity(
            collection_id=collection_id,
            entity_id=file_entity.entity_id,
            entity_type="file",
            content_hash=content_hash,
            entity_metadata={
                "title": file_entity.title,
                "file_path": file_entity.file_path,
                "file_type": file_entity.file_type,
                **(file_entity.metadata or {})
            }
        )
        db.add(db_entity)
        entities_inserted += 1

        # Chunk and embed
        chunk_texts = chunker.chunk(file_entity.content)
        chunks_created += len(chunk_texts)

        embeddings, provider_name = await router.embed(
            chunk_texts,
            required_dimension=collection.vector_dimension
        )

        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk = ChunkEntity(
                parent_id=file_entity.entity_id,
                chunk_index=i,
                content=text,
                title=file_entity.title,
                metadata=file_entity.metadata,
                embedding=embedding
            )
            all_chunk_entities.append(chunk)

        documents_processed += 1

    # Commit to database
    await db.commit()

    # Upsert to Qdrant
    if all_chunk_entities:
        await qdrant.upsert_chunks(str(collection_id), all_chunk_entities)

    processing_time_ms = int((time.time() - start_time) * 1000)

    return IngestResponse(
        collection_id=collection_id,
        documents_processed=documents_processed,
        chunks_created=chunks_created,
        entities_inserted=entities_inserted,
        entities_updated=entities_updated,
        processing_time_ms=processing_time_ms
    )


@router.post("/github", response_model=IngestResponse)
async def ingest_github(
    collection_id: uuid.UUID,
    request: GitHubIngestRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client),
    router: ProviderRouter = Depends(get_embedding_router)
):
    """Ingest documents from GitHub repository."""
    start_time = time.time()

    # Verify collection exists
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Initialize source
    source = GitHubSource()
    if not await source.validate_config():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub token not configured. Set GITHUB_TOKEN environment variable."
        )

    chunker = get_chunker()

    # Track stats
    documents_processed = 0
    chunks_created = 0
    entities_inserted = 0
    entities_updated = 0
    all_chunk_entities: List[ChunkEntity] = []

    # Fetch and process files
    async for file_entity in source.fetch(
        repo_name=request.repo_name,
        branch=request.branch,
        path=request.path,
        extensions=request.extensions
    ):
        # Compute content hash
        content_hash = hash_content(file_entity.content)

        # Check if entity exists
        result = await db.execute(
            select(Entity).where(
                Entity.collection_id == collection_id,
                Entity.content_hash == content_hash
            )
        )
        existing_entity = result.scalar_one_or_none()

        if existing_entity:
            entities_updated += 1
            continue

        # Create entity
        db_entity = Entity(
            collection_id=collection_id,
            entity_id=file_entity.entity_id,
            entity_type="file",
            content_hash=content_hash,
            entity_metadata={
                "title": file_entity.title,
                "file_path": file_entity.file_path,
                **(file_entity.metadata or {})
            }
        )
        db.add(db_entity)
        entities_inserted += 1

        # Chunk and embed
        chunk_texts = chunker.chunk(file_entity.content)
        chunks_created += len(chunk_texts)

        embeddings, provider_name = await router.embed(
            chunk_texts,
            required_dimension=collection.vector_dimension
        )

        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk = ChunkEntity(
                parent_id=file_entity.entity_id,
                chunk_index=i,
                content=text,
                title=file_entity.title,
                metadata=file_entity.metadata,
                embedding=embedding
            )
            all_chunk_entities.append(chunk)

        documents_processed += 1

    # Commit to database
    await db.commit()

    # Upsert to Qdrant
    if all_chunk_entities:
        await qdrant.upsert_chunks(str(collection_id), all_chunk_entities)

    processing_time_ms = int((time.time() - start_time) * 1000)

    return IngestResponse(
        collection_id=collection_id,
        documents_processed=documents_processed,
        chunks_created=chunks_created,
        entities_inserted=entities_inserted,
        entities_updated=entities_updated,
        processing_time_ms=processing_time_ms
    )
