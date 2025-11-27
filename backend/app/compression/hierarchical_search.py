"""Hierarchical search with selective expansion for compressed RAG.

Implements top-down search strategy:
1. Search document-level summaries first (fast, compressed)
2. Expand relevant documents to section level
3. Expand relevant sections to chunk level (original content)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.compression.hierarchical_summarization import SummaryLayer
from app.core.logging import get_logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.compression.models import HierarchicalSummary

logger = get_logger(__name__)


@dataclass
class HierarchicalSearchResult:
    """Result from hierarchical search with expansion path."""
    content: str
    score: float
    layer: SummaryLayer
    node_id: str
    parent_id: Optional[str]
    children_ids: List[str]
    expansion_path: List[str]  # Path from document → section → chunk
    metadata: Dict[str, Any]


class HierarchicalSearchEngine:
    """Top-down search with selective expansion."""

    def __init__(
        self,
        qdrant: QdrantStorage,
        provider: ProviderRouter,
        db: AsyncSession
    ):
        """Initialize hierarchical search engine.

        Args:
            qdrant: Qdrant storage client
            provider: Provider router for embeddings
            db: Database session
        """
        self.qdrant = qdrant
        self.provider = provider
        self.db = db

    async def search_hierarchical(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        top_k_documents: int = 3,
        top_k_sections_per_doc: int = 2,
        top_k_chunks_per_section: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HierarchicalSearchResult]:
        """Perform hierarchical search with selective expansion.

        Args:
            query: User query
            collection_id: Collection to search
            vector_dimension: Expected embedding dimension
            top_k_documents: Number of top documents to retrieve
            top_k_sections_per_doc: Sections to expand per document
            top_k_chunks_per_section: Chunks to expand per section
            filters: Optional Qdrant filters

        Returns:
            List of hierarchical search results with expansion paths
        """
        logger.info(
            "hierarchical_search_started",
            collection_id=collection_id,
            top_k_documents=top_k_documents
        )

        # Generate query embedding
        query_embeddings, _ = await self.provider.embed(
            [query],
            required_dimension=vector_dimension
        )
        query_embedding = query_embeddings[0]

        # Step 1: Search document-level summaries
        document_results = await self._search_layer(
            collection_id=collection_id,
            query_embedding=query_embedding,
            layer=SummaryLayer.DOCUMENT,
            limit=top_k_documents,
            filters=filters
        )

        if not document_results:
            logger.warning("no_documents_found", collection_id=collection_id)
            return []

        logger.info(
            "documents_retrieved",
            count=len(document_results),
            top_score=document_results[0]["score"] if document_results else 0
        )

        # Step 2: Expand top documents to sections
        expanded_results = []
        for doc_result in document_results:
            doc_node_id = doc_result["id"]
            doc_score = doc_result["score"]

            # Get document node from database
            doc_node = await self._get_summary_node(doc_node_id)
            if not doc_node:
                continue

            # Search sections that are children of this document
            section_filters = {
                **( filters or {}),
                "must": [
                    {
                        "key": "parent_id",
                        "match": {"value": doc_node_id}
                    }
                ]
            }

            section_results = await self._search_layer(
                collection_id=collection_id,
                query_embedding=query_embedding,
                layer=SummaryLayer.SECTION,
                limit=top_k_sections_per_doc,
                filters=section_filters
            )

            # Step 3: Expand top sections to chunks
            for section_result in section_results:
                section_node_id = section_result["id"]
                section_score = section_result["score"]

                section_node = await self._get_summary_node(section_node_id)
                if not section_node:
                    continue

                # Search chunks that are children of this section
                chunk_filters = {
                    **(filters or {}),
                    "must": [
                        {
                            "key": "parent_id",
                            "match": {"value": section_node_id}
                        }
                    ]
                }

                chunk_results = await self._search_layer(
                    collection_id=collection_id,
                    query_embedding=query_embedding,
                    layer=SummaryLayer.CHUNK,
                    limit=top_k_chunks_per_section,
                    filters=chunk_filters
                )

                # Create results with expansion path
                for chunk_result in chunk_results:
                    chunk_node = await self._get_summary_node(chunk_result["id"])
                    if not chunk_node:
                        continue

                    result = HierarchicalSearchResult(
                        content=chunk_node.content,
                        score=chunk_result["score"],
                        layer=SummaryLayer.CHUNK,
                        node_id=chunk_node.node_id,
                        parent_id=chunk_node.parent_id,
                        children_ids=chunk_node.children_ids,
                        expansion_path=[doc_node_id, section_node_id, chunk_node.node_id],
                        metadata=chunk_node.node_metadata or {}
                    )
                    expanded_results.append(result)

        # Sort by score
        expanded_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "hierarchical_search_completed",
            total_results=len(expanded_results),
            documents_expanded=len(document_results),
            compression_ratio=f"1:{top_k_documents * top_k_sections_per_doc * top_k_chunks_per_section}/{len(expanded_results)}"
        )

        return expanded_results

    async def _search_layer(
        self,
        collection_id: str,
        query_embedding: List[float],
        layer: SummaryLayer,
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search specific layer in hierarchy."""
        # Add layer filter
        layer_filters = {
            **(filters or {}),
            "must": [
                *((filters or {}).get("must", [])),
                {
                    "key": "layer",
                    "match": {"value": layer.value}
                }
            ]
        }

        results = await self.qdrant.search(
            collection_id=collection_id,
            query_vector=query_embedding,
            limit=limit,
            filters=layer_filters
        )

        return results

    async def _get_summary_node(self, node_id: str) -> Optional[HierarchicalSummary]:
        """Retrieve summary node from database."""
        result = await self.db.execute(
            select(HierarchicalSummary).where(HierarchicalSummary.node_id == node_id)
        )
        return result.scalar_one_or_none()


def get_hierarchical_search_engine(
    qdrant: QdrantStorage,
    provider: ProviderRouter,
    db: AsyncSession
) -> HierarchicalSearchEngine:
    """Factory function to get hierarchical search engine instance."""
    return HierarchicalSearchEngine(qdrant, provider, db)
