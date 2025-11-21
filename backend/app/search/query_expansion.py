"""Query expansion service using Ollama LLM."""

import json
from typing import List, Optional
from app.core.providers.ollama import OllamaProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class QueryExpansionService:
    """Service for expanding search queries using LLM."""

    def __init__(self, provider: Optional[OllamaProvider] = None):
        """Initialize query expansion service.

        Args:
            provider: Ollama provider instance (creates new one if not provided)
        """
        self.provider = provider or OllamaProvider()

    async def expand_query(self, query: str, num_variations: int = 4) -> List[str]:
        """Expand a search query into multiple variations.

        Args:
            query: Original user query
            num_variations: Number of query variations to generate (default: 4)

        Returns:
            List of query variations including the original query
        """
        system_prompt = """You are a search query expansion assistant. Given a user's search query, generate alternative phrasings and related queries that would help find relevant information.

Rules:
1. Generate diverse variations that capture different aspects of the query
2. Include synonyms, related terms, and different phrasings
3. Keep variations concise and focused
4. Return ONLY a JSON array of strings, no other text
5. Do not include the original query in your variations"""

        user_prompt = f"""Generate {num_variations} alternative search queries for: "{query}"

Return format (JSON array only):
["variation 1", "variation 2", "variation 3", "variation 4"]"""

        try:
            logger.debug(
                "query_expansion_starting",
                query=query,
                num_variations=num_variations,
                provider=self.provider.name,
                model=self.provider.llm_model
            )

            # Generate variations using Ollama
            response = await self.provider.generate(
                prompt=user_prompt,
                system=system_prompt
            )

            logger.debug(
                "query_expansion_response_received",
                response_length=len(response),
                response_preview=response[:200]
            )

            # Parse JSON response
            variations = self._parse_variations(response)

            logger.info(
                "query_expansion_success",
                query=query,
                variations_generated=len(variations)
            )

            # Ensure we have the requested number of variations
            variations = variations[:num_variations]

            # Always include the original query first
            return [query] + variations

        except Exception as e:
            # Fallback: return original query if expansion fails
            logger.error(
                "query_expansion_failed",
                query=query,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True  # This will log the full stack trace
            )
            print(f"Query expansion failed: {e}, using original query only")
            return [query]

    def _parse_variations(self, response: str) -> List[str]:
        """Parse LLM response to extract query variations.

        Args:
            response: Raw LLM response

        Returns:
            List of query variations
        """
        # Try to find JSON array in response
        response = response.strip()

        # Handle case where LLM wraps JSON in markdown code blocks
        if response.startswith("```"):
            # Extract content between ```json and ```
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
            variations = json.loads(response)
            if isinstance(variations, list):
                # Filter out empty strings and ensure all are strings
                return [str(v).strip() for v in variations if v and str(v).strip()]
            else:
                raise ValueError("Response is not a JSON array")
        except json.JSONDecodeError as e:
            # Fallback: try to extract queries line by line
            print(f"JSON parse failed: {e}, attempting line-by-line extraction")
            lines = response.split("\n")
            variations = []

            for line in lines:
                line = line.strip()
                # Skip empty lines and common prefixes
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                # Remove common list markers
                line = line.lstrip("-*•").strip()
                # Remove quotes if present
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                if line.startswith("'") and line.endswith("'"):
                    line = line[1:-1]

                if line:
                    variations.append(line)

            return variations

    async def close(self):
        """Close provider connections."""
        if self.provider:
            await self.provider.close()


# Singleton instance
_expansion_service: Optional[QueryExpansionService] = None


def get_expansion_service() -> QueryExpansionService:
    """Get or create query expansion service singleton.

    Returns:
        QueryExpansionService instance
    """
    global _expansion_service
    if _expansion_service is None:
        _expansion_service = QueryExpansionService()
    return _expansion_service
