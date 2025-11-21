"""Context window expansion for search results."""

from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.database import Entity
from app.storage.qdrant import QdrantStorage
from app.core.logging import get_logger

logger = get_logger(__name__)


async def expand_context_windows(
    results: List[Dict[str, Any]],
    db: AsyncSession,
    qdrant: QdrantStorage,
    collection_id: str,
    window_size: int = 1
) -> List[Dict[str, Any]]:
    """Expand search results by including surrounding chunks.

    For each result chunk, includes N chunks before and after to provide
    better context for synthesis and understanding.

    Args:
        results: Original search results
        db: Database session
        qdrant: Qdrant storage instance
        collection_id: Collection UUID
        window_size: Number of chunks to include on each side (default: 1)

    Returns:
        Results with expanded context in metadata
    """
    if not results or window_size < 1:
        return results

    logger.debug(
        "context_expansion_started",
        num_results=len(results),
        window_size=window_size
    )

    expanded_results = []

    for result in results:
        entity_id = result["entity_id"]
        chunk_index = result.get("metadata", {}).get("chunk_index")

        # If no chunk index, cannot expand - return as-is
        if chunk_index is None:
            expanded_results.append(result)
            continue

        # Get parent entity to find sibling chunks
        parent_id = entity_id.rsplit("_chunk_", 1)[0] if "_chunk_" in entity_id else None

        if not parent_id:
            expanded_results.append(result)
            continue

        # Calculate range of chunks to fetch
        start_idx = max(0, chunk_index - window_size)
        end_idx = chunk_index + window_size + 1  # +1 because range is exclusive

        # Query for surrounding chunks
        try:
            # Get chunks from Qdrant by filtering on parent_id and chunk_index range
            surrounding_chunks = await _get_chunks_by_index_range(
                qdrant,
                collection_id,
                parent_id,
                start_idx,
                end_idx
            )

            # Sort by chunk index
            surrounding_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))

            # Build context window
            context_before = []
            context_after = []

            for chunk in surrounding_chunks:
                chunk_idx = chunk.get("metadata", {}).get("chunk_index", 0)

                if chunk_idx < chunk_index:
                    context_before.append(chunk["content"])
                elif chunk_idx > chunk_index:
                    context_after.append(chunk["content"])

            # Add context to result metadata
            result_copy = result.copy()
            result_copy["metadata"] = result.get("metadata", {}).copy()
            result_copy["metadata"]["context_before"] = context_before
            result_copy["metadata"]["context_after"] = context_after
            result_copy["metadata"]["context_window_size"] = window_size

            # Optionally create expanded content
            expanded_content = []
            if context_before:
                expanded_content.append("... " + " ".join(context_before[-1:]))  # Last chunk before
            expanded_content.append(result["content"])
            if context_after:
                expanded_content.append(" ".join(context_after[:1]) + " ...")  # First chunk after

            result_copy["expanded_content"] = " ".join(expanded_content)

            expanded_results.append(result_copy)

            logger.debug(
                "context_expanded",
                entity_id=entity_id,
                chunk_index=chunk_index,
                chunks_before=len(context_before),
                chunks_after=len(context_after)
            )

        except Exception as e:
            logger.warning(
                "context_expansion_failed",
                entity_id=entity_id,
                chunk_index=chunk_index,
                error=str(e)
            )
            expanded_results.append(result)

    logger.info(
        "context_expansion_completed",
        original_count=len(results),
        expanded_count=len(expanded_results)
    )

    return expanded_results


async def _get_chunks_by_index_range(
    qdrant: QdrantStorage,
    collection_id: str,
    parent_id: str,
    start_idx: int,
    end_idx: int
) -> List[Dict[str, Any]]:
    """Get chunks within a specific index range.

    Args:
        qdrant: Qdrant storage
        collection_id: Collection ID
        parent_id: Parent entity ID
        start_idx: Start chunk index (inclusive)
        end_idx: End chunk index (exclusive)

    Returns:
        List of chunks in the range
    """
    # For now, we'll do a scroll and filter client-side
    # In production, you'd want to add proper filtering in Qdrant

    # FIXED: Add nexus_ prefix to collection name
    collection_name = f"nexus_{collection_id}"

    results, _ = await qdrant.client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "parent_id",
                    "match": {"value": parent_id}
                }
            ]
        },
        limit=100,  # Reasonable limit for sibling chunks
        with_payload=True,
        with_vectors=False
    )

    # Filter by chunk index range
    filtered = []
    for point in results:
        chunk_idx = point.payload.get("metadata", {}).get("chunk_index")
        if chunk_idx is not None and start_idx <= chunk_idx < end_idx:
            filtered.append({
                "entity_id": point.payload.get("entity_id"),
                "content": point.payload.get("content"),
                "metadata": point.payload.get("metadata", {})
            })

    return filtered
