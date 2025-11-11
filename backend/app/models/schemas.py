"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


# Collection Schemas
class CollectionCreate(BaseModel):
    """Create collection request."""
    name: str = Field(..., min_length=1, max_length=255)
    embedding_provider: str = Field(default="ollama", pattern="^(ollama|gemini|openai)$")
    vector_dimension: Optional[int] = Field(default=None, description="Auto-detected from provider if not specified")


class CollectionResponse(BaseModel):
    """Collection response."""
    id: UUID
    organization_id: UUID
    name: str
    embedding_provider: str
    vector_dimension: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Document/Entity Schemas
class DocumentCreate(BaseModel):
    """Document to ingest."""
    content: str = Field(..., min_length=1)
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    """Ingest documents request."""
    documents: List[DocumentCreate] = Field(..., min_items=1)


class IngestResponse(BaseModel):
    """Ingest response."""
    collection_id: UUID
    documents_processed: int
    chunks_created: int
    entities_inserted: int
    entities_updated: int
    processing_time_ms: int


# Search Schemas
class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    use_cache: bool = Field(default=True)
    expand_query: bool = Field(default=False, description="Use LLM to expand query into multiple variations")
    synthesize: bool = Field(default=False, description="Generate a synthesized answer from search results using LLM")
    hybrid: bool = Field(default=True, description="Use hybrid search (dense + BM25)")
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Individual search result."""
    entity_id: str
    content: str
    title: Optional[str] = None
    score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response."""
    collection_id: UUID
    query: str
    results: List[SearchResult]
    total_results: int
    latency_ms: int
    from_cache: bool
    provider_used: str
    expanded_queries: Optional[List[str]] = Field(default=None, description="Query variations used for search")
    synthesized_answer: Optional[str] = Field(default=None, description="LLM-generated answer from search results")
    tokens_used: Optional[int] = Field(default=None, description="Tokens used for synthesis")
    synthesis_cost_usd: Optional[float] = Field(default=None, description="Cost of synthesis operation")


# Analytics Schemas
class CollectionStats(BaseModel):
    """Collection statistics."""
    collection_id: UUID
    total_entities: int
    total_searches: int
    avg_search_latency_ms: float
    cache_hit_rate: float
    total_cost_usd: float
    last_search_at: Optional[datetime] = None


# Health Check
class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: Dict[str, str]


# Authentication
class APIKeyCreate(BaseModel):
    """Create API key request."""
    name: Optional[str] = None


class APIKeyResponse(BaseModel):
    """API key response (only returned on creation)."""
    id: UUID
    organization_id: UUID
    key: str  # Only returned on creation
    name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyListItem(BaseModel):
    """API key list item (without exposing the key)."""
    id: UUID
    name: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None

    class Config:
        from_attributes = True
