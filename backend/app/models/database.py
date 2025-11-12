"""SQLAlchemy database models."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean, Float, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.storage.postgres import Base


class Organization(Base):
    """Organization table for multi-tenancy."""

    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Relationships
    collections = relationship("Collection", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="organization", cascade="all, delete-orphan")


class APIKey(Base):
    """API keys for authentication."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="api_keys")


class Collection(Base):
    """Collection of documents from various sources."""

    __tablename__ = "collections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    embedding_provider = Column(String(50), nullable=False, default="ollama")  # ollama, gemini, openai
    vector_dimension = Column(Integer, nullable=False, default=768)  # 768, 1536, 3072
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationships
    organization = relationship("Organization", back_populates="collections")
    entities = relationship("Entity", back_populates="collection", cascade="all, delete-orphan")
    search_analytics = relationship("SearchAnalytics", back_populates="collection", cascade="all, delete-orphan")


class Entity(Base):
    """Entities (documents/chunks) tracked for change detection and source deduplication."""

    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("collections.id"), nullable=False)
    entity_id = Column(String(512), nullable=False)  # Source-specific ID
    entity_type = Column(String(50), nullable=False)  # file, email, page, etc.
    content_hash = Column(String(64), nullable=False)  # SHA256 hash for change detection
    entity_metadata = Column(JSON, nullable=True)  # Flexible metadata storage (renamed from metadata)

    # Source tracking for deduplication
    source_type = Column(String(50), nullable=True)  # "file_upload", "github", "gmail", etc.
    source_id = Column(String(512), nullable=True)  # Unique identifier from source (file path, commit hash, email ID)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationships
    collection = relationship("Collection", back_populates="entities")

    # Unique constraint
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )


class SearchAnalytics(Base):
    """Search analytics for tracking query performance and costs."""

    __tablename__ = "search_analytics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("collections.id"), nullable=False)
    query = Column(Text, nullable=False)
    result_count = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Integer, nullable=False)  # Milliseconds
    cache_hit = Column(Boolean, nullable=False, default=False)
    provider_used = Column(String(50), nullable=True)  # Which provider was used
    cost_usd = Column(Float, nullable=False, default=0.0)
    search_metadata = Column(JSON, nullable=True)  # Additional metadata (filters, etc.) - renamed from metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)

    # Relationships
    collection = relationship("Collection", back_populates="search_analytics")
    feedback = relationship("SearchFeedback", back_populates="search_analytics", cascade="all, delete-orphan")


class SearchFeedback(Base):
    """User feedback on search results for quality metrics."""

    __tablename__ = "search_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    search_analytics_id = Column(UUID(as_uuid=True), ForeignKey("search_analytics.id"), nullable=False)
    result_position = Column(Integer, nullable=False)  # Position in results (0-indexed)
    result_entity_id = Column(String(512), nullable=False)  # Which result was interacted with
    feedback_type = Column(String(50), nullable=False)  # "click", "copy", "thumbs_up", "thumbs_down"
    feedback_metadata = Column(JSON, nullable=True)  # Additional context (dwell_time, etc.)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Relationships
    search_analytics = relationship("SearchAnalytics", back_populates="feedback")
