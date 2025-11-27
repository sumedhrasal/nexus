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
from app.models.schemas import IngestRequest, IngestResponse, DirectoryIngestRequest, DocumentCreate
from app.api.dependencies import get_db, get_qdrant_client
from app.storage.qdrant import QdrantStorage
from app.core.chunking import get_chunker
# DISABLED: Parent-child chunking (keeping for future tuning)
# from app.core.semantic_chunking import get_semantic_chunker
from app.core.hashing import hash_content
from app.core.entities import ChunkEntity
from app.core.logging import get_logger
from app.core.metrics import record_ingestion_metrics
from app.core.metadata_extraction import get_metadata_extractor
from app.sources.directory_scanner import DirectoryScanner
from app.core.text_extraction import extract_text_from_html

router = APIRouter(prefix="/collections/{collection_id}/ingest", tags=["ingestion"])
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)


@router.post("", response_model=IngestResponse)
@limiter.limit("20/minute")  # 20 ingestion requests per minute
async def ingest_documents(
    request: Request,
    collection_id: uuid.UUID,
    ingest_request: IngestRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client)
):
    """Ingest documents into a collection."""
    start_time = time.time()

    logger.info(
        "ingestion_started",
        collection_id=str(collection_id),
        num_documents=len(ingest_request.documents)
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

    # Get provider router filtered to collection's embedding provider
    from app.api.dependencies import get_collection_router
    router = await get_collection_router(
        collection_id=collection_id,
        provider_override=ingest_request.provider,
        db=db
    )

    # Get chunker
    chunker = get_chunker()

    # Track stats
    documents_processed = 0
    chunks_created = 0
    entities_inserted = 0
    entities_updated = 0
    all_chunk_entities: List[ChunkEntity] = []

    # Get metadata extractor
    metadata_extractor = get_metadata_extractor()

    for doc in ingest_request.documents:
        # Compute content hash
        content_hash = hash_content(doc.content)

        # IMPROVEMENT: Extract metadata using LLM (incremental for large docs)
        extracted_metadata = {}
        try:
            extracted_metadata = await metadata_extractor.extract_document_metadata(
                content=doc.content,
                title=doc.title
            )
            logger.debug(
                "metadata_extracted",
                fields=list(extracted_metadata.keys()),
                has_title=bool(extracted_metadata.get("title"))
            )
        except Exception as e:
            logger.warning("metadata_extraction_failed", error=str(e), using_provided_metadata=True)

        # Check for source-level deduplication first (if source info provided)
        if doc.source_type and doc.source_id:
            source_result = await db.execute(
                select(Entity).where(
                    Entity.collection_id == collection_id,
                    Entity.source_type == doc.source_type,
                    Entity.source_id == doc.source_id
                )
            )
            existing_source_entity = source_result.scalar_one_or_none()

            if existing_source_entity:
                # Same source already ingested
                # Check if content changed (by comparing hash)
                if existing_source_entity.content_hash == content_hash:
                    # Content unchanged, skip
                    entities_updated += 1
                    logger.debug(
                        "source_dedup_skip",
                        source_type=doc.source_type,
                        source_id=doc.source_id,
                        reason="unchanged"
                    )
                    continue
                else:
                    # Content changed - need to update
                    # For now, we'll treat this as a new entity
                    # TODO: Implement update logic (delete old chunks, add new ones)
                    logger.info(
                        "source_content_changed",
                        source_type=doc.source_type,
                        source_id=doc.source_id
                    )

        # Check if entity exists by content hash (for non-source or new source content)
        result = await db.execute(
            select(Entity).where(
                Entity.collection_id == collection_id,
                Entity.content_hash == content_hash
            )
        )
        existing_entity = result.scalar_one_or_none()

        if existing_entity and not (doc.source_type and doc.source_id):
            # Content unchanged, skip (only if no source tracking)
            entities_updated += 1
            continue

        # Create or update entity
        entity_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Merge extracted metadata with provided metadata
        merged_metadata = {
            **(doc.metadata or {}),
            **extracted_metadata,  # LLM-extracted metadata
        }
        # Ensure provided title takes precedence
        if doc.title:
            merged_metadata["title"] = doc.title

        db_entity = Entity(
            collection_id=collection_id,
            entity_id=entity_id,
            entity_type="document",
            content_hash=content_hash,
            source_type=doc.source_type,
            source_id=doc.source_id,
            entity_metadata=merged_metadata
        )
        db.add(db_entity)
        entities_inserted += 1

        logger.debug(
            "entity_created",
            entity_id=entity_id,
            source_type=doc.source_type,
            source_id=doc.source_id
        )

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
                title=doc.title or extracted_metadata.get("title"),
                metadata=merged_metadata,  # Use merged metadata including LLM-extracted fields
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
    provider: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client)
):
    """Ingest a markdown file into a collection.

    Args:
        collection_id: UUID of the collection to ingest into
        file: Uploaded file (markdown format)
        title: Optional title for the document (defaults to filename)
        provider: Optional provider override (ollama, gemini, openai)

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

    # Get provider router filtered to collection's embedding provider
    from app.api.dependencies import get_collection_router
    router = await get_collection_router(
        collection_id=collection_id,
        provider_override=provider,
        db=db
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

    # Source tracking for file uploads
    source_type = "file_upload"
    source_id = file.filename  # Use filename as source ID

    # Check for source-level deduplication
    source_result = await db.execute(
        select(Entity).where(
            Entity.collection_id == collection_id,
            Entity.source_type == source_type,
            Entity.source_id == source_id
        )
    )
    existing_source_entity = source_result.scalar_one_or_none()

    documents_processed = 0
    chunks_created = 0
    entities_inserted = 0
    entities_updated = 0

    if existing_source_entity:
        # Same file already uploaded
        if existing_source_entity.content_hash == content_hash:
            # Content unchanged, skip
            entities_updated += 1
            logger.info(
                "file_already_ingested",
                filename=file.filename,
                collection_id=str(collection_id)
            )
    else:
        # IMPROVEMENT: Extract metadata using LLM
        metadata_extractor = get_metadata_extractor()
        extracted_metadata = {}
        try:
            extracted_metadata = await metadata_extractor.extract_document_metadata(
                content=content,
                title=doc_title
            )
            logger.debug(
                "metadata_extracted",
                filename=file.filename,
                fields=list(extracted_metadata.keys())
            )
        except Exception as e:
            logger.warning("metadata_extraction_failed", error=str(e), filename=file.filename)

        # Merge metadata
        merged_metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            **extracted_metadata
        }
        # Ensure provided title takes precedence
        if doc_title:
            merged_metadata["title"] = doc_title

        # Create entity with source tracking
        entity_id = f"doc_{uuid.uuid4().hex[:12]}"
        db_entity = Entity(
            collection_id=collection_id,
            entity_id=entity_id,
            entity_type="document",
            content_hash=content_hash,
            source_type=source_type,
            source_id=source_id,
            entity_metadata=merged_metadata
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
                title=doc_title or extracted_metadata.get("title"),
                metadata=merged_metadata,  # Use merged metadata including LLM-extracted fields
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


@router.post("/html", response_model=IngestResponse)
@limiter.limit("20/minute")  # 20 ingestion requests per minute
async def ingest_html(
    request: Request,
    collection_id: uuid.UUID,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client)
):
    """Ingest an HTML file into a collection.

    Args:
        collection_id: UUID of the collection to ingest into
        file: Uploaded HTML file
        title: Optional title for the document (defaults to filename)
        provider: Optional provider override (ollama, gemini, openai)

    Returns:
        IngestResponse with processing statistics

    Raises:
        HTTPException: If collection not found or file is invalid
    """
    start_time = time.time()

    logger.info(
        "html_ingestion_started",
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

    # Get provider router filtered to collection's embedding provider
    from app.api.dependencies import get_collection_router
    router = await get_collection_router(
        collection_id=collection_id,
        provider_override=provider,
        db=db
    )

    # Read file content
    try:
        content_bytes = await file.read()
        html_content = content_bytes.decode('utf-8')
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

    if not html_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content cannot be empty"
        )

    # Extract clean text from HTML
    try:
        content = extract_text_from_html(html_content, preserve_structure=True)
        logger.info(
            "html_text_extracted",
            filename=file.filename,
            html_size=len(html_content),
            text_size=len(content),
            reduction_pct=f"{(1 - len(content)/len(html_content))*100:.1f}%"
        )
    except Exception as e:
        logger.warning(
            "html_extraction_fallback",
            filename=file.filename,
            error=str(e)
        )
        # Fallback to using HTML as-is if extraction fails
        content = html_content

    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Extracted content cannot be empty"
        )

    # Use filename as title if not provided
    doc_title = title or file.filename or "Untitled Document"

    # Compute content hash
    content_hash = hash_content(content)

    # Source tracking for HTML file uploads
    source_type = "html_upload"
    source_id = file.filename  # Use filename as source ID

    # Check for source-level deduplication
    source_result = await db.execute(
        select(Entity).where(
            Entity.collection_id == collection_id,
            Entity.source_type == source_type,
            Entity.source_id == source_id
        )
    )
    existing_source_entity = source_result.scalar_one_or_none()

    documents_processed = 0
    chunks_created = 0
    entities_inserted = 0
    entities_updated = 0

    if existing_source_entity:
        # Same file already uploaded
        if existing_source_entity.content_hash == content_hash:
            # Content unchanged, skip
            entities_updated += 1
            logger.info(
                "html_already_ingested",
                filename=file.filename,
                collection_id=str(collection_id)
            )
    else:
        # IMPROVEMENT: Extract metadata using LLM
        metadata_extractor = get_metadata_extractor()
        extracted_metadata = {}
        try:
            extracted_metadata = await metadata_extractor.extract_document_metadata(
                content=content,
                title=doc_title
            )
            logger.debug(
                "metadata_extracted",
                filename=file.filename,
                fields=list(extracted_metadata.keys())
            )
        except Exception as e:
            logger.warning("metadata_extraction_failed", error=str(e), filename=file.filename)

        # Merge metadata
        html_merged_metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            **extracted_metadata
        }
        # Ensure provided title takes precedence
        if doc_title:
            html_merged_metadata["title"] = doc_title

        # Create entity with source tracking
        entity_id = f"doc_{uuid.uuid4().hex[:12]}"
        db_entity = Entity(
            collection_id=collection_id,
            entity_id=entity_id,
            entity_type="document",
            content_hash=content_hash,
            source_type=source_type,
            source_id=source_id,
            entity_metadata=html_merged_metadata
        )
        db.add(db_entity)
        entities_inserted += 1

        # DISABLED: Parent-child chunking (for future tuning)
        # Use simple token-aware chunking instead
        from app.core.chunking import get_chunker
        chunker = get_chunker(smart=True)  # SmartChunker preserves paragraphs
        chunk_texts = chunker.chunk(content)
        chunks_created += len(chunk_texts)

        logger.info(
            "simple_chunking_applied",
            filename=file.filename,
            total_chunks=len(chunk_texts),
            chunking_strategy="simple"
        )

        # Embed chunks
        embeddings, provider_name = await router.embed(
            chunk_texts,
            required_dimension=collection.vector_dimension
        )

        # Create chunk entities (simple chunking, no parent-child)
        all_chunk_entities: List[ChunkEntity] = []
        for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk = ChunkEntity(
                parent_id=entity_id,
                chunk_index=i,
                content=chunk_text,
                title=doc_title or extracted_metadata.get("title"),
                metadata=html_merged_metadata,  # Use merged metadata including LLM-extracted fields
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
        "html_ingestion_completed",
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


@router.post("/directory", response_model=IngestResponse)
@limiter.limit("10/hour")  # Stricter limit for directory scans (resource intensive)
async def ingest_directory(
    request: Request,
    collection_id: uuid.UUID,
    directory_request: DirectoryIngestRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStorage = Depends(get_qdrant_client)
):
    """Ingest files from a local directory (e.g., cloned git repository).

    This endpoint scans a local directory and ingests text files based on filters.
    Useful for ingesting cloned GitHub repositories or any local file collections.

    Args:
        collection_id: UUID of the collection to ingest into
        request: Directory ingestion configuration with path and filters

    Returns:
        IngestResponse with processing statistics

    Raises:
        HTTPException: If directory doesn't exist or too many files found
    """
    start_time = time.time()

    logger.info(
        "directory_ingestion_started",
        collection_id=str(collection_id),
        directory_path=directory_request.directory_path,
        file_types=directory_request.file_types,
        max_files=directory_request.max_files
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

    # Get provider router filtered to collection's embedding provider
    from app.api.dependencies import get_collection_router
    router = await get_collection_router(
        collection_id=collection_id,
        provider_override=directory_request.provider,
        db=db
    )

    # Initialize directory scanner
    try:
        scanner = DirectoryScanner(
            base_path=directory_request.directory_path,
            file_types=directory_request.file_types,
            include_paths=directory_request.include_paths,
            exclude_paths=directory_request.exclude_paths,
            max_file_size_mb=directory_request.max_file_size_mb
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Scan directory for files
    files = scanner.scan(max_files=directory_request.max_files)

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files found matching the specified filters"
        )

    if len(files) > directory_request.max_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Found {len(files)} files, but max_files is {directory_request.max_files}. "
                   "Adjust filters or increase max_files."
        )

    logger.info(
        "directory_scan_completed",
        files_found=len(files),
        directory_path=directory_request.directory_path
    )

    # Extract git metadata if requested
    git_metadata = None
    source_type = "directory"

    if directory_request.extract_git_metadata:
        git_metadata = scanner.extract_git_metadata()
        if git_metadata:
            source_type = "github"

    # Process each file and create DocumentCreate objects
    documents = []
    skipped_files = 0

    for file_info in files:
        try:
            # Read file content
            content = file_info["path"].read_text(encoding="utf-8")

            if not content.strip():
                logger.debug(
                    "empty_file_skipped",
                    path=str(file_info["relative_path"])
                )
                skipped_files += 1
                continue

            # Build source_id
            if git_metadata:
                # Format: "{repo_name}/{relative_path}@{commit_sha}"
                source_id = f"{git_metadata['repo_name']}/{file_info['relative_path']}@{git_metadata['commit_sha']}"
            else:
                # Format: "{relative_path}"
                source_id = str(file_info["relative_path"])

            # Create document with metadata
            metadata = {
                "path": str(file_info["relative_path"]),
                "size_mb": file_info["size_mb"],
            }

            # Add git metadata if available
            if git_metadata:
                metadata.update(git_metadata)

            documents.append(DocumentCreate(
                content=content,
                title=file_info["path"].name,
                metadata=metadata,
                source_type=source_type,
                source_id=source_id
            ))

        except UnicodeDecodeError:
            logger.warning(
                "file_encoding_error",
                path=str(file_info["relative_path"]),
                message="Failed to decode file as UTF-8"
            )
            skipped_files += 1
            continue
        except Exception as e:
            logger.warning(
                "file_read_error",
                path=str(file_info["relative_path"]),
                error=str(e)
            )
            skipped_files += 1
            continue

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No valid documents to ingest. {skipped_files} files were skipped due to errors."
        )

    logger.info(
        "documents_prepared",
        total_documents=len(documents),
        skipped_files=skipped_files
    )

    # Use existing ingestion logic
    ingest_req = IngestRequest(documents=documents)
    response = await ingest_documents(
        request=request,
        collection_id=collection_id,
        ingest_request=ingest_req,
        db=db,
        qdrant=qdrant,
        router=router
    )

    logger.info(
        "directory_ingestion_completed",
        collection_id=str(collection_id),
        directory_path=directory_request.directory_path,
        documents_processed=response.documents_processed,
        chunks_created=response.chunks_created,
        entities_inserted=response.entities_inserted,
        entities_updated=response.entities_updated,
        processing_time_ms=response.processing_time_ms
    )

    return response
