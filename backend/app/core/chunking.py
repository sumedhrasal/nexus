"""Text chunking with token-aware semantic splitting."""

import tiktoken
from typing import List
from app.config import settings


class SemanticChunker:
    """Token-aware chunker that preserves semantic boundaries."""

    def __init__(
        self,
        max_tokens: int = None,
        overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        """Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk (default: from settings)
            overlap: Token overlap between chunks (default: from settings)
            encoding_name: Tiktoken encoding name
        """
        self.max_tokens = max_tokens or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> List[str]:
        """Chunk text into overlapping segments.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Tokenize entire text
        tokens = self.tokenizer.encode(text)

        # If text fits in one chunk, return as-is
        if len(tokens) <= self.max_tokens:
            return [text]

        # Create overlapping chunks
        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move to next chunk with overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - self.overlap

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))


class SmartChunker:
    """Advanced chunker that tries to preserve semantic units (paragraphs, sentences)."""

    def __init__(
        self,
        max_tokens: int = None,
        overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        """Initialize smart chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks
            encoding_name: Tiktoken encoding name
        """
        self.max_tokens = max_tokens or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.fallback_chunker = SemanticChunker(max_tokens, overlap, encoding_name)

    def chunk(self, text: str) -> List[str]:
        """Chunk text preserving paragraph/sentence boundaries.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Try paragraph-based chunking first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds max, fall back to token chunking
            if para_tokens > self.max_tokens:
                # Flush current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                # Chunk this large paragraph
                para_chunks = self.fallback_chunker.chunk(para)
                chunks.extend(para_chunks)
                continue

            # Check if adding this paragraph would exceed limit
            if current_tokens + para_tokens > self.max_tokens:
                # Flush current chunk
                chunks.append(current_chunk.strip())
                current_chunk = para
                current_tokens = para_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))


# Default chunker instance
_default_chunker = None


def get_chunker(smart: bool = True) -> SemanticChunker:
    """Get default chunker instance.

    Args:
        smart: Use SmartChunker if True, else SemanticChunker

    Returns:
        Chunker instance
    """
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = SmartChunker() if smart else SemanticChunker()
    return _default_chunker
