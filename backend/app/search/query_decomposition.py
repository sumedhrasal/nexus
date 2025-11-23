"""Query decomposition service for complex queries.

This module uses an LLM to decompose complex queries into simpler sub-queries
that can be searched and synthesized independently.
"""

import json
from typing import List, Optional
from app.core.providers.router import ProviderRouter
from app.search.plan_metrics import log_decomposition_result
from app.core.logging import get_logger

logger = get_logger(__name__)


class QueryDecomposer:
    """Decomposes complex queries into sub-queries using LLM."""

    def __init__(self, provider: Optional[ProviderRouter] = None):
        """Initialize query decomposer.

        Args:
            provider: LLM provider for decomposition (creates new if not provided)
        """
        self.provider = provider or ProviderRouter(strategy="cost")  # Use cheapest LLM

    async def decompose(
        self,
        query: str,
        num_sub_queries: int = 3,
        context: Optional[str] = None
    ) -> List[str]:
        """Decompose query into sub-queries.

        Args:
            query: Original complex query
            num_sub_queries: Number of sub-queries to generate (2-5)
            context: Optional context about why decomposition is needed

        Returns:
            List of sub-queries

        Raises:
            Exception: If decomposition fails (caller should handle)
        """
        logger.info(
            "decomposition_started",
            query=query[:100],
            num_sub_queries=num_sub_queries
        )

        try:
            # Generate sub-queries using LLM
            sub_queries = await self._decompose_with_llm(
                query,
                num_sub_queries,
                context
            )

            logger.info(
                "decomposition_completed",
                original_query=query[:100],
                num_generated=len(sub_queries),
                sub_queries=[sq[:50] for sq in sub_queries]
            )

            # Log decomposition result for metrics
            log_decomposition_result(
                query=query,
                sub_queries=sub_queries,
                num_generated=len(sub_queries)
            )

            return sub_queries

        except Exception as e:
            logger.error(
                "decomposition_failed",
                error=str(e),
                query=query[:100],
                exc_info=True
            )
            raise

    async def _decompose_with_llm(
        self,
        query: str,
        num_sub_queries: int,
        context: Optional[str]
    ) -> List[str]:
        """Generate sub-queries using LLM.

        Args:
            query: Original query
            num_sub_queries: Number of sub-queries to generate
            context: Optional context

        Returns:
            List of sub-queries

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        system_prompt = self._build_decomposition_system_prompt()
        user_prompt = self._build_decomposition_user_prompt(
            query,
            num_sub_queries,
            context
        )

        logger.debug(
            "llm_decomposition_request",
            system_prompt_length=len(system_prompt),
            user_prompt_length=len(user_prompt)
        )

        # Generate sub-queries with LLM
        response, provider_name = await self.provider.generate(
            prompt=user_prompt,
            system=system_prompt
        )

        logger.debug(
            "llm_decomposition_response",
            provider=provider_name,
            response_length=len(response)
        )

        # Parse response
        sub_queries = self._parse_decomposition_response(response, num_sub_queries)

        return sub_queries

    def _build_decomposition_system_prompt(self) -> str:
        """Build system prompt for decomposition LLM."""
        return """You are a query decomposition expert for a search system.

Your job is to break down complex queries into simpler, focused sub-queries that can be searched independently and then synthesized into a comprehensive answer.

You must return ONLY a valid JSON object with this EXACT structure:
{
  "sub_queries": [
    "sub-query 1",
    "sub-query 2",
    "sub-query 3"
  ],
  "reasoning": "brief explanation of decomposition strategy (1-2 sentences)"
}

GUIDELINES:

**When to Decompose:**
- Multi-faceted questions (compare, contrast, analyze multiple aspects)
- Questions with multiple subjects or concepts
- Questions asking for comprehensive coverage ("explain all...", "what are the different...")
- Comparison questions ("X vs Y", "differences between A and B")

**Decomposition Strategies:**
1. **Aspect Decomposition**: Break into different aspects/dimensions
   - Original: "Compare X and Y"
   - Sub-queries: ["What is X?", "What is Y?", "What are the key differences between X and Y?"]

2. **Temporal Decomposition**: Break into time periods
   - Original: "History of X"
   - Sub-queries: ["Early development of X", "Modern evolution of X", "Current state of X"]

3. **Hierarchical Decomposition**: Break into general → specific
   - Original: "How does X work?"
   - Sub-queries: ["What is X?", "Core components of X", "Step-by-step process of X"]

4. **Multi-Subject Decomposition**: Separate distinct subjects
   - Original: "Explain A, B, and C"
   - Sub-queries: ["What is A?", "What is B?", "What is C?"]

**Quality Guidelines:**
- Each sub-query should be self-contained and searchable independently
- Sub-queries should be complementary (cover different aspects, minimal overlap)
- Keep sub-queries simple and focused (avoid creating complex sub-queries)
- Maintain the intent and context of the original query
- Number of sub-queries should match requested count (typically 2-5)

**Important:**
- Return ONLY valid JSON, no markdown, no extra text
- Sub-queries should be in natural language (not keywords)
- Ensure sub-queries together cover the full scope of original query"""

    def _build_decomposition_user_prompt(
        self,
        query: str,
        num_sub_queries: int,
        context: Optional[str]
    ) -> str:
        """Build user prompt for decomposition LLM.

        Args:
            query: Original query
            num_sub_queries: Number of sub-queries to generate
            context: Optional context

        Returns:
            Formatted user prompt
        """
        context_note = f"\n\nContext: {context}" if context else ""

        return f"""Decompose this query into {num_sub_queries} focused sub-queries:

Original Query: "{query}"{context_note}

Generate exactly {num_sub_queries} sub-queries that:
1. Are simpler and more focused than the original
2. Cover different aspects of the original query
3. Can be searched independently
4. Together provide comprehensive coverage

Return the JSON object as specified in the system prompt."""

    def _parse_decomposition_response(
        self,
        response: str,
        expected_count: int
    ) -> List[str]:
        """Parse LLM response to extract sub-queries.

        Args:
            response: Raw LLM response
            expected_count: Expected number of sub-queries

        Returns:
            List of sub-queries

        Raises:
            ValueError: If response is invalid
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
                if in_code_block or not line.strip().startswith("```"):
                    json_lines.append(line)

            response = "\n".join(json_lines).strip()

        # Parse JSON
        try:
            data = json.loads(response)

            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data)}")

            if "sub_queries" not in data:
                raise ValueError("Missing 'sub_queries' field in response")

            sub_queries = data["sub_queries"]

            if not isinstance(sub_queries, list):
                raise ValueError(f"Expected list for sub_queries, got {type(sub_queries)}")

            if len(sub_queries) == 0:
                raise ValueError("No sub-queries generated")

            # Filter out empty strings
            sub_queries = [sq.strip() for sq in sub_queries if sq and sq.strip()]

            if len(sub_queries) == 0:
                raise ValueError("All sub-queries were empty")

            # Validate count (allow some flexibility)
            if len(sub_queries) < 2:
                raise ValueError(f"Too few sub-queries generated: {len(sub_queries)}")

            if len(sub_queries) > expected_count + 1:
                logger.warning(
                    "too_many_sub_queries",
                    expected=expected_count,
                    generated=len(sub_queries),
                    truncating=True
                )
                sub_queries = sub_queries[:expected_count]

            logger.debug(
                "decomposition_parsed",
                num_sub_queries=len(sub_queries),
                reasoning=data.get("reasoning", "")[:100]
            )

            return sub_queries

        except json.JSONDecodeError as e:
            logger.error(
                "decomposition_json_parse_failed",
                error=str(e),
                response_preview=response[:200],
                exc_info=True
            )
            raise ValueError(f"Failed to parse decomposition JSON: {e}")

    async def close(self):
        """Close provider connections."""
        if self.provider:
            # ProviderRouter doesn't have close method currently
            # but we keep this for future compatibility
            pass


# Singleton instance
_decomposer: Optional[QueryDecomposer] = None


def get_query_decomposer(
    provider: Optional[ProviderRouter] = None
) -> QueryDecomposer:
    """Get or create query decomposer singleton.

    Args:
        provider: Optional provider override

    Returns:
        QueryDecomposer instance
    """
    global _decomposer

    if _decomposer is None:
        _decomposer = QueryDecomposer(provider=provider)

    return _decomposer
