"""Qdrant vector database client."""

import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)

from app.config import settings
from app.core.entities import ChunkEntity
import logging

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Qdrant vector database client for hybrid search."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Qdrant client.

        Args:
            url: Qdrant server URL (default: from settings)
            api_key: Qdrant API key (default: from settings)
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.client = AsyncQdrantClient(url=self.url, api_key=self.api_key, timeout=30.0)

    async def create_collection(
        self,
        collection_id: str,
        vector_dimension: int
    ):
        """Create collection with dense and sparse vectors.

        Args:
            collection_id: Collection UUID as string
            vector_dimension: Dimension of dense vectors (768, 1536, 3072)
        """
        collection_name = f"nexus_{collection_id}"

        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            if collection_name in [c.name for c in collections.collections]:
                logger.info(f"Collection {collection_name} already exists")
                return

            # Create collection with dense + sparse vectors
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_dimension,
                        distance=Distance.COSINE
                    )
                },
                # sparse_vectors_config={
                #     "bm25": {}  # Sparse BM25 vectors for keyword search
                # }
            )

            logger.info(f"Created collection {collection_name} with dimension {vector_dimension}")

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

    async def upsert_chunks(
        self,
        collection_id: str,
        chunks: List[ChunkEntity]
    ):
        """Upsert chunk embeddings to collection.

        Args:
            collection_id: Collection UUID as string
            chunks: List of chunk entities with embeddings
        """
        collection_name = f"nexus_{collection_id}"

        try:
            points = []
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.chunk_id} missing embedding, skipping")
                    continue

                # Build payload with parent-child support
                payload = {
                    "entity_id": chunk.parent_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "title": chunk.title,
                    "metadata": chunk.metadata or {}
                }

                # Add parent-child hierarchy fields if present
                # Note: Keep parent_content moderate to avoid HTTP timeouts
                if hasattr(chunk, 'parent_content') and chunk.parent_content:
                    # Truncate parent content to avoid HTTP write timeouts
                    parent_content = chunk.parent_content
                    max_parent_size = 5000  # ~5KB limit (conservative for batch uploads)
                    if len(parent_content) > max_parent_size:
                        logger.debug(
                            f"Parent content truncated from {len(parent_content)} to {max_parent_size} chars"
                        )
                        parent_content = parent_content[:max_parent_size] + "... [truncated]"
                    payload["parent_content"] = parent_content

                if hasattr(chunk, 'parent_chunk_id') and chunk.parent_chunk_id:
                    payload["parent_chunk_id"] = chunk.parent_chunk_id
                if hasattr(chunk, 'is_child_chunk'):
                    payload["is_child_chunk"] = chunk.is_child_chunk

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                    vector={"dense": chunk.embedding},
                    payload=payload
                )
                points.append(point)

            if points:
                await self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                logger.info(f"Upserted {len(points)} chunks to {collection_name}")

        except Exception as e:
            logger.error(
                f"Failed to upsert chunks to {collection_name}: {e}",
                exc_info=True,
                extra={
                    "collection_name": collection_name,
                    "num_chunks": len(chunks),
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
            raise

    async def search(
        self,
        collection_id: str,
        query_vector: List[float],
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search collection with dense vector.

        Args:
            collection_id: Collection UUID as string
            query_vector: Query embedding vector
            limit: Maximum results to return
            offset: Offset for pagination
            filters: Optional filters (future)

        Returns:
            List of search results with scores
        """
        collection_name = f"nexus_{collection_id}"

        try:
            # Simple dense vector search for now
            results = await self.client.search(
                collection_name=collection_name,
                query_vector=("dense", query_vector),
                limit=limit + offset,  # Get extra for offset
                with_payload=True,
                with_vectors=False
            )

            # Apply offset manually
            results = results[offset:offset + limit]

            # Format results
            formatted = []
            for result in results:
                result_dict = {
                    "entity_id": result.payload.get("entity_id"),
                    "chunk_id": result.payload.get("chunk_id"),
                    "content": result.payload.get("content"),
                    "title": result.payload.get("title"),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {})
                }

                # Include parent-child fields if present
                if "parent_content" in result.payload:
                    result_dict["parent_content"] = result.payload["parent_content"]
                if "parent_chunk_id" in result.payload:
                    result_dict["parent_chunk_id"] = result.payload["parent_chunk_id"]
                if "is_child_chunk" in result.payload:
                    result_dict["is_child_chunk"] = result.payload["is_child_chunk"]

                formatted.append(result_dict)

            return formatted

        except Exception as e:
            logger.error(f"Search failed for {collection_name}: {e}")
            raise

    async def delete_collection(self, collection_id: str):
        """Delete entire collection.

        Args:
            collection_id: Collection UUID as string
        """
        collection_name = f"nexus_{collection_id}"

        try:
            await self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise

    async def count_points(self, collection_id: str) -> int:
        """Count points in collection.

        Args:
            collection_id: Collection UUID as string

        Returns:
            Number of points
        """
        collection_name = f"nexus_{collection_id}"

        try:
            info = await self.client.get_collection(collection_name=collection_name)
            # Handle different qdrant-client versions
            if hasattr(info, 'points_count'):
                return info.points_count
            elif hasattr(info, 'vectors_count'):
                return info.vectors_count
            elif isinstance(info, dict):
                return info.get('points_count', info.get('vectors_count', 0))
            else:
                logger.warning(f"Unknown collection info format: {type(info)}")
                return 0
        except Exception as e:
            logger.error(f"Failed to count points in {collection_name}: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False


# Global instance
_qdrant_client: Optional[QdrantStorage] = None


def get_qdrant() -> QdrantStorage:
    """Get global Qdrant client instance.

    Returns:
        QdrantStorage instance
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantStorage()
    return _qdrant_client
