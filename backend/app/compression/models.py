"""Database models for compression features."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.storage.postgres import Base
from app.compression.hierarchical_summarization import SummaryLayer


class HierarchicalSummary(Base):
    """Hierarchical summary nodes for RAPTOR-style compression.

    Stores the 3-layer hierarchy:
    - Layer 0: Original chunks
    - Layer 1: Section summaries
    - Layer 2: Document summaries
    """

    __tablename__ = "hierarchical_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("collections.id"), nullable=False)
    entity_id = Column(String(512), ForeignKey("entities.entity_id"), nullable=False)

    # Hierarchical node info
    node_id = Column(String(512), unique=True, nullable=False, index=True)
    layer = Column(SQLEnum(SummaryLayer), nullable=False, index=True)
    content = Column(Text, nullable=False)

    # Tree structure
    parent_id = Column(String(512), nullable=True, index=True)  # null for layer 2 (document)
    children_ids = Column(JSON, nullable=False, default=list)  # List of child node_ids

    # Original chunk reference (only for layer 0)
    original_chunk_index = Column(Integer, nullable=True)

    # Metadata
    node_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Note: Embeddings stored in Qdrant with node_id as ID
    # This allows us to search across all layers and expand selectively


class KnowledgeGraphNode(Base):
    """Knowledge graph nodes for entity extraction.

    Stores extracted entities and their relationships.
    """

    __tablename__ = "knowledge_graph_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("collections.id"), nullable=False)
    entity_id = Column(String(512), ForeignKey("entities.entity_id"), nullable=False)

    # Entity info
    entity_name = Column(String(512), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False, index=True)  # PERSON, ORG, CONCEPT, etc.
    description = Column(Text, nullable=True)

    # Source chunks (where entity was mentioned)
    source_chunk_ids = Column(JSON, nullable=False, default=list)

    # Metadata
    node_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)


class KnowledgeGraphEdge(Base):
    """Knowledge graph edges for relationships between entities."""

    __tablename__ = "knowledge_graph_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("collections.id"), nullable=False)

    # Relationship
    source_node_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_graph_nodes.id"), nullable=False, index=True)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_graph_nodes.id"), nullable=False, index=True)
    relationship_type = Column(String(100), nullable=False, index=True)  # "mentions", "related_to", etc.

    # Source context (chunk where relationship was found)
    source_chunk_id = Column(String(512), nullable=True)

    # Metadata
    edge_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
