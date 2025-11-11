"""Document ingestion endpoints."""

import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.database import Collection, Entity
from app.models.schemas import IngestRequest, IngestResponse
from app.api.dependencies import get_db, get_qdrant_client, get_embedding_router
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.core.chunking import get_chunker
from app.core.hashing import hash_content
from app.core.entities import ChunkEntity
from app.core.logging import get_logger
from app.core.metrics import record_ingestion_metrics

router = APIRouter(prefix="/collections/{collection_id}/ingest", tags=["ingestion"])
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)


@router.post("", response_model=IngestResponse)
@limiter.limit("20/minute")  # 20 ingestion requests per minute
async def ingest_documents(
    request_obj: Request,
    collection_id: uuid.UUID,
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client),
    router: ProviderRouter = Depends(get_embedding_router)
):
    """Ingest documents into a collection."""
    start_time = time.time()

    logger.info(
        "ingestion_started",
        collection_id=str(collection_id),
        num_documents=len(request.documents)
    )

    # Verify collection exists
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        logger.warning("collection_not_found", collection_id=str(collection_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Get chunker
    chunker = get_chunker()

    # Track stats
    documents_processed = 0
    chunks_created = 0
    entities_inserted = 0
    entities_updated = 0
    all_chunk_entities: List[ChunkEntity] = []

    for doc in request.documents:
        # Compute content hash
        content_hash = hash_content(doc.content)

        # Check if entity exists
        result = await db.execute(
            select(Entity).where(
                Entity.collection_id == collection_id,
                Entity.content_hash == content_hash
            )
        )
        existing_entity = result.scalar_one_or_none()

        if existing_entity:
            # Content unchanged, skip
            entities_updated += 1
            continue

        # Create or update entity
        entity_id = f"doc_{uuid.uuid4().hex[:12]}"
        db_entity = Entity(
            collection_id=collection_id,
            entity_id=entity_id,
            entity_type="document",
            content_hash=content_hash,
            entity_metadata={
                "title": doc.title,
                **(doc.metadata or {})
            }
        )
        db.add(db_entity)
        entities_inserted += 1

        # Chunk content
        chunk_texts = chunker.chunk(doc.content)
        chunks_created += len(chunk_texts)

        # Embed chunks
        embeddings, provider_name = await router.embed(
            chunk_texts,
            required_dimension=collection.vector_dimension
        )

        # Create chunk entities
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk = ChunkEntity(
                parent_id=entity_id,
                chunk_index=i,
                content=text,
                title=doc.title,
                metadata=doc.metadata,
                embedding=embedding
            )
            all_chunk_entities.append(chunk)

        documents_processed += 1

    # Commit entities to database
    await db.commit()

    # Upsert all chunks to Qdrant in batch
    if all_chunk_entities:
        await qdrant.upsert_chunks(str(collection_id), all_chunk_entities)

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Record metrics
    record_ingestion_metrics(
        collection_id=str(collection_id),
        duration_seconds=(time.time() - start_time),
        documents_processed=documents_processed,
        chunks_created=chunks_created
    )

    logger.info(
        "ingestion_completed",
        collection_id=str(collection_id),
        documents_processed=documents_processed,
        chunks_created=chunks_created,
        entities_inserted=entities_inserted,
        entities_updated=entities_updated,
        processing_time_ms=processing_time_ms
    )

    return IngestResponse(
        collection_id=collection_id,
        documents_processed=documents_processed,
        chunks_created=chunks_created,
        entities_inserted=entities_inserted,
        entities_updated=entities_updated,
        processing_time_ms=processing_time_ms
    )


@router.post("/file", response_model=IngestResponse)
@limiter.limit("20/minute")  # 20 ingestion requests per minute
async def ingest_file(
    request: Request,
    collection_id: uuid.UUID,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client),
    router: ProviderRouter = Depends(get_embedding_router)
):
    """Ingest a markdown file into a collection.

    Args:
        collection_id: UUID of the collection to ingest into
        file: Uploaded file (markdown format)
        title: Optional title for the document (defaults to filename)

    Returns:
        IngestResponse with processing statistics

    Raises:
        HTTPException: If collection not found or file is invalid
    """
    start_time = time.time()

    logger.info(
        "file_ingestion_started",
        collection_id=str(collection_id),
        filename=file.filename
    )

    # Verify collection exists
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        logger.warning("collection_not_found", collection_id=str(collection_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found"
        )

    # Read file content
    try:
        content_bytes = await file.read()
        content = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be valid UTF-8 encoded text"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading file: {str(e)}"
        )

    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content cannot be empty"
        )

    # Use filename as title if not provided
    doc_title = title or file.filename or "Untitled Document"

    # Compute content hash
    content_hash = hash_content(content)

    # Check if entity exists
    result = await db.execute(
        select(Entity).where(
            Entity.collection_id == collection_id,
            Entity.content_hash == content_hash
        )
    )
    existing_entity = result.scalar_one_or_none()

    documents_processed = 0
    chunks_created = 0
    entities_inserted = 0
    entities_updated = 0

    if existing_entity:
        # Content unchanged, skip
        entities_updated += 1
    else:
        # Create entity
        entity_id = f"doc_{uuid.uuid4().hex[:12]}"
        db_entity = Entity(
            collection_id=collection_id,
            entity_id=entity_id,
            entity_type="document",
            content_hash=content_hash,
            entity_metadata={
                "title": doc_title,
                "filename": file.filename,
                "content_type": file.content_type
            }
        )
        db.add(db_entity)
        entities_inserted += 1

        # Get chunker
        chunker = get_chunker()

        # Chunk content
        chunk_texts = chunker.chunk(content)
        chunks_created += len(chunk_texts)

        # Embed chunks
        embeddings, provider_name = await router.embed(
            chunk_texts,
            required_dimension=collection.vector_dimension
        )

        # Create chunk entities
        all_chunk_entities: List[ChunkEntity] = []
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk = ChunkEntity(
                parent_id=entity_id,
                chunk_index=i,
                content=text,
                title=doc_title,
                metadata={
                    "filename": file.filename,
                    "content_type": file.content_type
                },
                embedding=embedding
            )
            all_chunk_entities.append(chunk)

        # Commit entities to database
        await db.commit()

        # Upsert all chunks to Qdrant in batch
        if all_chunk_entities:
            await qdrant.upsert_chunks(str(collection_id), all_chunk_entities)

        documents_processed += 1

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Record metrics
    record_ingestion_metrics(
        collection_id=str(collection_id),
        duration_seconds=(time.time() - start_time),
        documents_processed=documents_processed,
        chunks_created=chunks_created
    )

    logger.info(
        "file_ingestion_completed",
        collection_id=str(collection_id),
        filename=file.filename,
        documents_processed=documents_processed,
        chunks_created=chunks_created,
        entities_inserted=entities_inserted,
        entities_updated=entities_updated,
        processing_time_ms=processing_time_ms
    )

    return IngestResponse(
        collection_id=collection_id,
        documents_processed=documents_processed,
        chunks_created=chunks_created,
        entities_inserted=entities_inserted,
        entities_updated=entities_updated,
        processing_time_ms=processing_time_ms
    )
