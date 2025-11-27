"""Graph-augmented search with selective decompression.

Implements the compression + queryability strategy:
1. Search graph summaries (high-level, compressed)
2. Traverse graph to find related concepts
3. Selectively decompress only visited nodes' content
4. Synthesize answer from graph context
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.compression.knowledge_graph import KnowledgeGraphStorage
from app.compression.entity_extraction import EntityExtractor
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GraphSearchResult:
    """Result from graph-augmented search."""
    content: str
    score: float
    entity_name: str
    entity_type: str
    source_chunk_ids: List[str]
    graph_context: Dict[str, Any]  # Related entities and relationships
    expansion_path: List[str]  # Entities visited during graph traversal


class GraphSearchEngine:
    """Graph-augmented search with selective decompression."""

    def __init__(
        self,
        qdrant: QdrantStorage,
        provider: ProviderRouter,
        db: AsyncSession
    ):
        """Initialize graph search engine.

        Args:
            qdrant: Qdrant storage client
            provider: Provider router for embeddings and LLM
            db: Database session
        """
        self.qdrant = qdrant
        self.provider = provider
        self.db = db
        self.kg_storage = KnowledgeGraphStorage(db)
        self.entity_extractor = EntityExtractor(provider, use_llm=True)

    async def search_with_graph(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        top_k_entities: int = 5,
        graph_depth: int = 2,
        final_limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[GraphSearchResult]:
        """Perform graph-augmented search.

        Steps:
        1. Extract key entities from query
        2. Search for relevant entities in graph
        3. Traverse graph to find related concepts
        4. Selectively decompress visited chunks
        5. Return graph-augmented results

        Args:
            query: User query
            collection_id: Collection to search
            vector_dimension: Expected embedding dimension
            top_k_entities: Number of top entities to start traversal from
            graph_depth: Maximum graph traversal depth
            final_limit: Final number of results to return
            filters: Optional Qdrant filters

        Returns:
            List of graph-augmented search results
        """
        logger.info(
            "graph_search_started",
            collection_id=collection_id,
            top_k_entities=top_k_entities,
            graph_depth=graph_depth
        )

        # Step 1: Extract key entities from query
        query_entities = await self.entity_extractor.extract_entities(
            text=query,
            chunk_id="query",
            max_entities=5
        )

        if not query_entities:
            logger.warning("no_query_entities_found", falling_back_to_semantic_search=True)
            # Fallback to regular semantic search
            return await self._fallback_semantic_search(
                query, collection_id, vector_dimension, final_limit, filters
            )

        logger.info(
            "query_entities_extracted",
            entities=[e.name for e in query_entities]
        )

        # Step 2: Traverse graph from query entities
        entity_names = [e.name for e in query_entities]
        visited_nodes, visited_chunk_ids = await self.kg_storage.traverse_graph(
            start_entity_names=entity_names,
            collection_id=uuid.UUID(collection_id),
            max_depth=graph_depth,
            max_nodes=top_k_entities * 3
        )

        if not visited_nodes:
            logger.warning("graph_traversal_empty", falling_back_to_semantic_search=True)
            return await self._fallback_semantic_search(
                query, collection_id, vector_dimension, final_limit, filters
            )

        logger.info(
            "graph_traversal_completed",
            visited_nodes=len(visited_nodes),
            visited_chunks=len(visited_chunk_ids)
        )

        # Step 3: Selectively decompress visited chunks
        # Search only in the chunks identified by graph traversal
        chunk_filter = {
            **(filters or {}),
            "must": [
                *((filters or {}).get("must", [])),
                {
                    "key": "node_id",
                    "match": {"any": list(visited_chunk_ids)[:100]}  # Limit for performance
                }
            ]
        }

        # Generate query embedding
        query_embeddings, _ = await self.provider.embed(
            [query],
            required_dimension=vector_dimension
        )

        # Search selectively decompressed chunks
        search_results = await self.qdrant.search(
            collection_id=collection_id,
            query_vector=query_embeddings[0],
            limit=final_limit,
            filters=chunk_filter
        )

        # Step 4: Build graph context for each result
        graph_results = []
        for result in search_results:
            # Find associated graph nodes
            associated_nodes = [
                node for node in visited_nodes
                if result["id"] in node.source_chunk_ids
            ]

            if associated_nodes:
                # Build graph context
                graph_context = {
                    "entities": [
                        {
                            "name": node.entity_name,
                            "type": node.entity_type,
                            "description": node.description
                        }
                        for node in associated_nodes
                    ],
                    "expansion_depth": graph_depth,
                    "total_visited_nodes": len(visited_nodes)
                }

                graph_result = GraphSearchResult(
                    content=result["payload"].get("content", ""),
                    score=result["score"],
                    entity_name=associated_nodes[0].entity_name,
                    entity_type=associated_nodes[0].entity_type,
                    source_chunk_ids=[result["id"]],
                    graph_context=graph_context,
                    expansion_path=[node.entity_name for node in associated_nodes]
                )
                graph_results.append(graph_result)

        logger.info(
            "graph_search_completed",
            total_results=len(graph_results),
            compression_ratio=f"{len(visited_chunk_ids)}/{final_limit}"
        )

        return graph_results[:final_limit]

    async def _fallback_semantic_search(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[GraphSearchResult]:
        """Fallback to semantic search when graph search fails."""
        query_embeddings, _ = await self.provider.embed(
            [query],
            required_dimension=vector_dimension
        )

        results = await self.qdrant.search(
            collection_id=collection_id,
            query_vector=query_embeddings[0],
            limit=limit,
            filters=filters
        )

        # Convert to GraphSearchResult format
        graph_results = []
        for result in results:
            graph_result = GraphSearchResult(
                content=result["payload"].get("content", ""),
                score=result["score"],
                entity_name="Unknown",
                entity_type="Unknown",
                source_chunk_ids=[result["id"]],
                graph_context={"fallback": True},
                expansion_path=[]
            )
            graph_results.append(graph_result)

        return graph_results


def get_graph_search_engine(
    qdrant: QdrantStorage,
    provider: ProviderRouter,
    db: AsyncSession
) -> GraphSearchEngine:
    """Factory function to get graph search engine instance."""
    return GraphSearchEngine(qdrant, provider, db)
