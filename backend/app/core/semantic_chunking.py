"""Semantic chunking with parent-child strategy for optimal retrieval."""

from typing import List, Tuple, Dict, Any
import re
from app.core.logging import get_logger

logger = get_logger(__name__)


class SemanticChunker:
    """Parent-child chunking strategy for precise retrieval with rich context.

    Strategy:
    1. Split text into semantic parent chunks (paragraphs/sections)
    2. Further split each parent into small child chunks (sentences/clauses)
    3. Index child chunks for precise vector search
    4. Retrieve parent chunks for LLM context
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,  # ~500 tokens
        child_chunk_size: int = 400,     # ~100 tokens
        parent_overlap: int = 200,       # 10% overlap
        child_overlap: int = 40          # 10% overlap
    ):
        """Initialize semantic chunker.

        Args:
            parent_chunk_size: Target size for parent chunks (chars)
            child_chunk_size: Target size for child chunks (chars)
            parent_overlap: Overlap between parent chunks (chars)
            child_overlap: Overlap between child chunks (chars)
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap

    def detect_semantic_boundaries(self, text: str) -> List[int]:
        """Detect semantic boundaries in text (paragraphs, sections).

        Args:
            text: Input text

        Returns:
            List of boundary positions (character indices)
        """
        boundaries = [0]  # Start of text

        # Pattern 1: Double newlines (paragraph breaks)
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append(match.end())

        # Pattern 2: Section headers (lines starting with #, or all caps)
        for match in re.finditer(r'\n(?:[#]{1,6}\s+|[A-Z\s]{10,})\n', text):
            boundaries.append(match.start())

        # Pattern 3: List boundaries (numbered/bulleted lists)
        for match in re.finditer(r'\n(?:\d+\.|[-*•])\s+', text):
            if match.start() not in boundaries:
                boundaries.append(match.start())

        boundaries.append(len(text))  # End of text
        boundaries = sorted(set(boundaries))  # Remove duplicates, sort

        return boundaries

    def create_parent_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """Create parent chunks using semantic boundaries.

        Args:
            text: Input text

        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        boundaries = self.detect_semantic_boundaries(text)
        parent_chunks = []

        current_chunk = []
        current_start = 0
        current_size = 0

        for i in range(len(boundaries) - 1):
            segment_start = boundaries[i]
            segment_end = boundaries[i + 1]
            segment = text[segment_start:segment_end].strip()
            segment_size = len(segment)

            # Skip empty segments
            if not segment:
                continue

            # If adding this segment exceeds target size and we have content
            if current_size + segment_size > self.parent_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ''.join(current_chunk).strip()
                if chunk_text:
                    parent_chunks.append((chunk_text, current_start, segment_start))

                # Start new chunk with overlap
                overlap_start = max(0, segment_start - self.parent_overlap)
                current_chunk = [text[overlap_start:segment_end]]
                current_start = overlap_start
                current_size = len(current_chunk[0])
            else:
                # Add segment to current chunk
                if not current_chunk:
                    current_start = segment_start
                current_chunk.append(segment + '\n')
                current_size += segment_size

        # Add final chunk
        if current_chunk:
            chunk_text = ''.join(current_chunk).strip()
            if chunk_text:
                parent_chunks.append((chunk_text, current_start, len(text)))

        return parent_chunks

    def create_child_chunks(self, text: str, start_offset: int = 0) -> List[Tuple[str, int, int]]:
        """Create small child chunks from text using sentence boundaries.

        Args:
            text: Input text (typically a parent chunk)
            start_offset: Starting position in original document

        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        child_chunks = []
        current_chunk = []
        current_start = start_offset
        current_size = 0
        position = start_offset

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            # If adding this sentence exceeds target and we have content
            if current_size + sentence_size > self.child_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                child_chunks.append((chunk_text, current_start, position))

                # Start new chunk with last sentence for overlap
                if self.child_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text, sentence]
                    current_start = position - len(overlap_text)
                else:
                    current_chunk = [sentence]
                    current_start = position

                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

            position += sentence_size + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            child_chunks.append((chunk_text, current_start, position))

        return child_chunks

    def chunk_with_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """Create hierarchical chunks (parent-child structure).

        Args:
            text: Input text

        Returns:
            List of chunk dictionaries with metadata
        """
        parent_chunks = self.create_parent_chunks(text)

        all_chunks = []
        parent_index = 0

        for parent_text, parent_start, parent_end in parent_chunks:
            # Create parent chunk metadata
            parent_id = f"parent_{parent_index}"

            # Create child chunks from this parent
            child_chunks = self.create_child_chunks(parent_text, parent_start)

            child_index = 0
            for child_text, child_start, child_end in child_chunks:
                chunk_metadata = {
                    'content': child_text,
                    'parent_content': parent_text,
                    'parent_id': parent_id,
                    'parent_index': parent_index,
                    'child_index': child_index,
                    'is_parent': False,
                    'start_pos': child_start,
                    'end_pos': child_end,
                    'parent_start': parent_start,
                    'parent_end': parent_end,
                    'chunk_type': 'child'
                }
                all_chunks.append(chunk_metadata)
                child_index += 1

            logger.debug(
                "parent_chunk_created",
                parent_index=parent_index,
                num_children=child_index,
                parent_size=len(parent_text),
                avg_child_size=len(parent_text) // max(1, child_index)
            )

            parent_index += 1

        logger.info(
            "hierarchical_chunking_completed",
            total_parents=parent_index,
            total_children=len(all_chunks),
            avg_children_per_parent=len(all_chunks) / max(1, parent_index)
        )

        return all_chunks


def get_semantic_chunker() -> SemanticChunker:
    """Get semantic chunker instance.

    Returns:
        SemanticChunker instance
    """
    return SemanticChunker()
