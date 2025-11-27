"""Knowledge graph storage and traversal for graph-augmented search.

Stores extracted entities and relationships in PostgreSQL.
Enables graph traversal to find related concepts and selective decompression.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import uuid

from app.compression.models import KnowledgeGraphNode, KnowledgeGraphEdge
from app.compression.entity_extraction import ExtractedEntity, ExtractedRelationship
from app.core.logging import get_logger

logger = get_logger(__name__)


class KnowledgeGraphStorage:
    """Store and retrieve knowledge graph nodes and edges."""

    def __init__(self, db: AsyncSession):
        """Initialize knowledge graph storage.

        Args:
            db: Database session
        """
        self.db = db

    async def store_entities(
        self,
        entities: List[ExtractedEntity],
        collection_id: uuid.UUID,
        entity_id: str
    ) -> List[uuid.UUID]:
        """Store extracted entities as graph nodes.

        Args:
            entities: Extracted entities
            collection_id: Collection UUID
            entity_id: Parent entity ID

        Returns:
            List of created node UUIDs
        """
        node_ids = []

        for entity in entities:
            # Check if entity already exists (by name and type)
            result = await self.db.execute(
                select(KnowledgeGraphNode).where(
                    and_(
                        KnowledgeGraphNode.collection_id == collection_id,
                        KnowledgeGraphNode.entity_name == entity.name,
                        KnowledgeGraphNode.entity_type == entity.entity_type.value
                    )
                )
            )
            existing_node = result.scalar_one_or_none()

            if existing_node:
                # Update existing node - add source chunk
                if entity.source_chunk_id not in existing_node.source_chunk_ids:
                    existing_node.source_chunk_ids.append(entity.source_chunk_id)
                    await self.db.commit()
                node_ids.append(existing_node.id)
            else:
                # Create new node
                node = KnowledgeGraphNode(
                    collection_id=collection_id,
                    entity_id=entity_id,
                    entity_name=entity.name,
                    entity_type=entity.entity_type.value,
                    description=entity.description,
                    source_chunk_ids=[entity.source_chunk_id],
                    node_metadata={
                        "confidence": entity.confidence,
                        "mentions": entity.mentions
                    }
                )
                self.db.add(node)
                await self.db.flush()
                node_ids.append(node.id)

        await self.db.commit()
        logger.info("entities_stored", count=len(node_ids), new=len(entities) - len(node_ids))

        return node_ids

    async def store_relationships(
        self,
        relationships: List[ExtractedRelationship],
        collection_id: uuid.UUID
    ) -> int:
        """Store extracted relationships as graph edges.

        Args:
            relationships: Extracted relationships
            collection_id: Collection UUID

        Returns:
            Number of edges created
        """
        edges_created = 0

        for rel in relationships:
            # Find source and target nodes
            source_result = await self.db.execute(
                select(KnowledgeGraphNode).where(
                    and_(
                        KnowledgeGraphNode.collection_id == collection_id,
                        KnowledgeGraphNode.entity_name == rel.source_entity
                    )
                )
            )
            source_node = source_result.scalar_one_or_none()

            target_result = await self.db.execute(
                select(KnowledgeGraphNode).where(
                    and_(
                        KnowledgeGraphNode.collection_id == collection_id,
                        KnowledgeGraphNode.entity_name == rel.target_entity
                    )
                )
            )
            target_node = target_result.scalar_one_or_none()

            if not source_node or not target_node:
                continue

            # Check if edge already exists
            edge_result = await self.db.execute(
                select(KnowledgeGraphEdge).where(
                    and_(
                        KnowledgeGraphEdge.source_node_id == source_node.id,
                        KnowledgeGraphEdge.target_node_id == target_node.id,
                        KnowledgeGraphEdge.relationship_type == rel.relationship_type
                    )
                )
            )
            existing_edge = edge_result.scalar_one_or_none()

            if not existing_edge:
                # Create edge
                edge = KnowledgeGraphEdge(
                    collection_id=collection_id,
                    source_node_id=source_node.id,
                    target_node_id=target_node.id,
                    relationship_type=rel.relationship_type,
                    source_chunk_id=rel.source_chunk_id,
                    edge_metadata={
                        "confidence": rel.confidence,
                        "context": rel.context
                    }
                )
                self.db.add(edge)
                edges_created += 1

        await self.db.commit()
        logger.info("relationships_stored", count=edges_created)

        return edges_created

    async def traverse_graph(
        self,
        start_entity_names: List[str],
        collection_id: uuid.UUID,
        max_depth: int = 2,
        max_nodes: int = 50
    ) -> Tuple[List[KnowledgeGraphNode], Set[str]]:
        """Traverse knowledge graph from starting entities.

        Args:
            start_entity_names: Entity names to start traversal from
            collection_id: Collection UUID
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to return

        Returns:
            Tuple of (visited nodes, visited chunk IDs for decompression)
        """
        visited_nodes = []
        visited_chunk_ids = set()
        queue = []

        # Find starting nodes
        for name in start_entity_names:
            result = await self.db.execute(
                select(KnowledgeGraphNode).where(
                    and_(
                        KnowledgeGraphNode.collection_id == collection_id,
                        KnowledgeGraphNode.entity_name == name
                    )
                )
            )
            node = result.scalar_one_or_none()
            if node:
                queue.append((node, 0))  # (node, depth)

        visited_node_ids = set()

        # BFS traversal
        while queue and len(visited_nodes) < max_nodes:
            node, depth = queue.pop(0)

            if node.id in visited_node_ids:
                continue

            visited_node_ids.add(node.id)
            visited_nodes.append(node)
            visited_chunk_ids.update(node.source_chunk_ids)

            # Stop if max depth reached
            if depth >= max_depth:
                continue

            # Find connected nodes
            # Outgoing edges
            outgoing_result = await self.db.execute(
                select(KnowledgeGraphEdge).where(
                    KnowledgeGraphEdge.source_node_id == node.id
                )
            )
            for edge in outgoing_result.scalars():
                target_result = await self.db.execute(
                    select(KnowledgeGraphNode).where(
                        KnowledgeGraphNode.id == edge.target_node_id
                    )
                )
                target_node = target_result.scalar_one_or_none()
                if target_node and target_node.id not in visited_node_ids:
                    queue.append((target_node, depth + 1))

            # Incoming edges
            incoming_result = await self.db.execute(
                select(KnowledgeGraphEdge).where(
                    KnowledgeGraphEdge.target_node_id == node.id
                )
            )
            for edge in incoming_result.scalars():
                source_result = await self.db.execute(
                    select(KnowledgeGraphNode).where(
                        KnowledgeGraphNode.id == edge.source_node_id
                    )
                )
                source_node = source_result.scalar_one_or_none()
                if source_node and source_node.id not in visited_node_ids:
                    queue.append((source_node, depth + 1))

        logger.info(
            "graph_traversal_completed",
            start_entities=len(start_entity_names),
            visited_nodes=len(visited_nodes),
            visited_chunks=len(visited_chunk_ids)
        )

        return visited_nodes, visited_chunk_ids


def get_knowledge_graph_storage(db: AsyncSession) -> KnowledgeGraphStorage:
    """Factory function to get knowledge graph storage instance."""
    return KnowledgeGraphStorage(db)
