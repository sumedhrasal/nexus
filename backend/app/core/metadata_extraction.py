"""Incremental metadata extraction for documents using LLM."""

from typing import Dict, Any, List, Optional
import json
from app.core.providers.ollama import OllamaProvider
from app.core.text_extraction import estimate_token_count, truncate_to_token_limit
from app.core.logging import get_logger
from app.config import settings

logger = get_logger(__name__)


class MetadataExtractor:
    """Extract metadata from documents using incremental/streaming approach."""

    def __init__(self, provider: Optional[OllamaProvider] = None):
        """Initialize metadata extractor.

        Args:
            provider: Optional Ollama provider for LLM-based extraction
        """
        self.provider = provider or OllamaProvider()
        self.context_window = settings.ollama_context_window

    async def extract_document_metadata(
        self,
        content: str,
        title: Optional[str] = None,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract metadata from document using incremental processing.

        For large documents, processes in chunks and merges results.

        Args:
            content: Document content
            title: Optional document title
            chunk_size: Optional chunk size (defaults to context window)

        Returns:
            Dictionary containing extracted metadata
        """
        chunk_size = chunk_size or (self.context_window - 500)  # Leave room for prompt

        # Estimate tokens
        estimated_tokens = estimate_token_count(content)

        logger.debug(
            "metadata_extraction_started",
            content_length=len(content),
            estimated_tokens=estimated_tokens,
            chunk_size=chunk_size,
            title=title
        )

        # If content fits in one chunk, extract directly
        if estimated_tokens <= chunk_size:
            return await self._extract_from_chunk(content, title, is_complete=True)

        # Otherwise, process incrementally
        return await self._extract_incremental(content, title, chunk_size)

    async def _extract_from_chunk(
        self,
        content: str,
        title: Optional[str],
        is_complete: bool = False
    ) -> Dict[str, Any]:
        """Extract metadata from a single chunk.

        Args:
            content: Content chunk
            title: Optional title
            is_complete: Whether this is the complete document

        Returns:
            Extracted metadata
        """
        system_prompt = """You are a metadata extraction assistant. Analyze the provided text and extract key metadata.

Return ONLY a JSON object with these fields (all optional):
{
  "title": "document title if found",
  "authors": ["author1", "author2"],
  "date": "publication date if found",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "summary": "brief 1-2 sentence summary",
  "document_type": "research_paper|article|documentation|report|email|etc",
  "topics": ["main topic 1", "main topic 2"]
}"""

        context_note = "complete document" if is_complete else "document excerpt"
        user_prompt = f"""Extract metadata from this {context_note}:

Title: {title or 'Unknown'}

Content:
{content[:2000]}{"..." if len(content) > 2000 else ""}

Return only the JSON object."""

        try:
            response = await self.provider.generate(
                prompt=user_prompt,
                system=system_prompt
            )

            # Parse JSON response
            metadata = self._parse_metadata_response(response)

            # Add title if not extracted
            if not metadata.get("title") and title:
                metadata["title"] = title

            logger.info(
                "metadata_extracted_from_chunk",
                fields_extracted=list(metadata.keys()),
                is_complete=is_complete
            )

            return metadata

        except Exception as e:
            logger.error(
                "metadata_extraction_failed",
                error=str(e),
                exc_info=True
            )
            # Return minimal metadata
            return {"title": title} if title else {}

    async def _extract_incremental(
        self,
        content: str,
        title: Optional[str],
        chunk_size: int
    ) -> Dict[str, Any]:
        """Extract metadata incrementally from large document.

        Strategy:
        1. Extract from beginning (abstract/intro)
        2. Extract from middle sections
        3. Extract from end (conclusion)
        4. Merge all metadata

        Args:
            content: Full document content
            title: Optional title
            chunk_size: Size of each chunk to process

        Returns:
            Merged metadata from all chunks
        """
        logger.info(
            "incremental_extraction_started",
            content_length=len(content),
            chunk_size=chunk_size
        )

        all_metadata: List[Dict[str, Any]] = []

        # Strategy: Extract from key sections
        # 1. Beginning (30% of chunk size)
        beginning = truncate_to_token_limit(content, int(chunk_size * 0.3))
        beginning_meta = await self._extract_from_chunk(beginning, title, is_complete=False)
        all_metadata.append(beginning_meta)

        # 2. Middle (if document is long enough)
        if len(content) > chunk_size * 2:
            middle_start = len(content) // 2 - int(chunk_size * 0.15)
            middle_end = len(content) // 2 + int(chunk_size * 0.15)
            middle = content[middle_start:middle_end]
            middle_meta = await self._extract_from_chunk(middle, title, is_complete=False)
            all_metadata.append(middle_meta)

        # 3. End (30% of chunk size)
        end = content[-int(chunk_size * 0.3):]
        end_meta = await self._extract_from_chunk(end, title, is_complete=False)
        all_metadata.append(end_meta)

        # Merge metadata
        merged = self._merge_metadata(all_metadata)

        logger.info(
            "incremental_extraction_completed",
            chunks_processed=len(all_metadata),
            merged_fields=list(merged.keys())
        )

        return merged

    def _merge_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge metadata from multiple chunks.

        Args:
            metadata_list: List of metadata dictionaries

        Returns:
            Merged metadata dictionary
        """
        merged: Dict[str, Any] = {}

        for metadata in metadata_list:
            for key, value in metadata.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, list):
                    # Merge lists, deduplicate
                    if isinstance(merged[key], list):
                        merged[key] = list(set(merged[key] + value))
                    else:
                        merged[key] = value
                elif isinstance(value, str) and value:
                    # Keep first non-empty string value
                    if not merged.get(key):
                        merged[key] = value

        return merged

    def _parse_metadata_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract metadata JSON.

        Args:
            response: Raw LLM response

        Returns:
            Parsed metadata dictionary
        """
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)

            response = "\n".join(json_lines).strip()

        # Parse JSON
        try:
            metadata = json.loads(response)
            if isinstance(metadata, dict):
                return metadata
            else:
                logger.warning("metadata_not_dict", response=response[:200])
                return {}
        except json.JSONDecodeError as e:
            logger.warning(
                "metadata_json_parse_failed",
                error=str(e),
                response=response[:200]
            )
            return {}

    async def extract_chunk_metadata(self, chunk: str) -> Dict[str, Any]:
        """Extract metadata from a single chunk (simpler, faster).

        Args:
            chunk: Text chunk

        Returns:
            Chunk-level metadata
        """
        system_prompt = """Extract key information from this text chunk. Return JSON with:
{
  "section_type": "abstract|introduction|methods|results|conclusion|discussion|other",
  "key_concepts": ["concept1", "concept2"],
  "entities": ["entity1", "entity2"]
}"""

        user_prompt = f"""Analyze this text chunk:

{chunk[:1000]}

Return only JSON."""

        try:
            response = await self.provider.generate(
                prompt=user_prompt,
                system=system_prompt
            )

            return self._parse_metadata_response(response)

        except Exception as e:
            logger.debug("chunk_metadata_extraction_failed", error=str(e))
            return {}

    async def close(self):
        """Close provider connections."""
        if self.provider:
            await self.provider.close()


# Singleton instance
_extractor: Optional[MetadataExtractor] = None


def get_metadata_extractor() -> MetadataExtractor:
    """Get or create metadata extractor singleton.

    Returns:
        MetadataExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = MetadataExtractor()
    return _extractor
