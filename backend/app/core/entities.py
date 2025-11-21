"""Entity data classes for document processing."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class BaseEntity:
    """Base entity representing a document or data object."""

    entity_id: str  # Unique ID from source system
    entity_type: str  # Type: file, email, page, etc.
    content: str  # Main text content
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Set defaults."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class ChunkEntity:
    """Chunk of a parent entity for embedding.

    Supports hierarchical parent-child chunking:
    - Child chunks: small, precise chunks for vector search
    - Parent chunks: larger context chunks for LLM synthesis
    """

    parent_id: str  # Parent entity ID
    chunk_index: int  # Index of this chunk
    content: str  # Chunk text
    title: Optional[str] = None  # Inherited from parent
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # Dense embedding
    sparse_embedding: Optional[Dict[str, Any]] = None  # Sparse BM25 embedding

    # Parent-child hierarchy fields
    parent_content: Optional[str] = None  # Full parent chunk content
    parent_chunk_id: Optional[str] = None  # ID of parent chunk
    is_child_chunk: bool = False  # True if this is a child of a larger parent

    @property
    def chunk_id(self) -> str:
        """Get unique chunk ID."""
        return f"{self.parent_id}_chunk_{self.chunk_index}"


@dataclass
class FileEntity(BaseEntity):
    """File-based entity."""

    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None

    def __post_init__(self):
        """Set entity type."""
        super().__post_init__()
        self.entity_type = "file"


@dataclass
class EmailEntity(BaseEntity):
    """Email entity."""

    sender: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    thread_id: Optional[str] = None

    def __post_init__(self):
        """Set entity type."""
        super().__post_init__()
        self.entity_type = "email"
        if self.subject and not self.title:
            self.title = self.subject


@dataclass
class PageEntity(BaseEntity):
    """Web page or wiki page entity."""

    url: Optional[str] = None
    author: Optional[str] = None

    def __post_init__(self):
        """Set entity type."""
        super().__post_init__()
        self.entity_type = "page"
