"""Query reformulation for better search results."""

from typing import List, Optional
import re
from app.core.providers.ollama import OllamaProvider
from app.core.logging import get_logger
from app.config import settings

logger = get_logger(__name__)


class QueryReformulator:
    """Reformulate queries to improve search effectiveness.

    Uses LLM when available, falls back to rule-based reformulation.
    """

    def __init__(self, provider: Optional[OllamaProvider] = None):
        """Initialize query reformulator.

        Args:
            provider: Optional Ollama provider for LLM-based reformulation
        """
        self.provider = provider or OllamaProvider()

    async def reformulate(self, query: str, use_llm: bool = True) -> str:
        """Reformulate query to be more search-friendly.

        Converts natural language questions into keyword-rich search queries.

        Args:
            query: Original user query
            use_llm: Whether to attempt LLM-based reformulation

        Returns:
            Reformulated query optimized for search
        """
        if use_llm:
            try:
                return await self._llm_reformulate(query)
            except Exception as e:
                logger.warning(
                    "llm_reformulation_failed",
                    query=query,
                    error=str(e),
                    fallback="rule_based"
                )

        # Fallback to rule-based reformulation
        return self._rule_based_reformulate(query)

    async def _llm_reformulate(self, query: str) -> str:
        """Use LLM to reformulate query.

        Args:
            query: Original query

        Returns:
            LLM-reformulated query
        """
        system_prompt = """You are a search query optimizer. Your job is to convert natural language questions into effective keyword-based search queries.

Rules:
1. Remove question words (what, how, why, when, where, who)
2. Extract key concepts and entities
3. Add relevant synonyms or related terms
4. Keep it concise (max 10 words)
5. Return ONLY the reformulated query, no explanations"""

        user_prompt = f"""Reformulate this search query to be more effective:
"{query}"

Return only the optimized search query."""

        try:
            response = await self.provider.generate(
                prompt=user_prompt,
                system=system_prompt
            )

            reformulated = response.strip().strip('"\'')

            logger.info(
                "llm_reformulation_success",
                original=query,
                reformulated=reformulated
            )

            return reformulated

        except Exception as e:
            logger.error(
                "llm_reformulation_error",
                query=query,
                error=str(e),
                exc_info=True
            )
            raise

    def _rule_based_reformulate(self, query: str) -> str:
        """Use rule-based approach to reformulate query.

        This works without LLM by applying linguistic patterns.

        Args:
            query: Original query

        Returns:
            Rule-reformulated query
        """
        reformulated = query.lower()

        # Remove question words at the start
        question_patterns = [
            r'^what\s+(is|are|was|were)\s+',
            r'^how\s+(do|does|did|can|could|would|should)\s+',
            r'^why\s+(is|are|was|were|do|does|did)\s+',
            r'^when\s+(is|are|was|were|do|does|did)\s+',
            r'^where\s+(is|are|was|were|do|does|did)\s+',
            r'^who\s+(is|are|was|were)\s+',
            r'^which\s+',
            r'^can\s+you\s+(tell|explain|describe)\s+(me\s+)?',
            r'^tell\s+me\s+(about\s+)?',
            r'^explain\s+',
            r'^describe\s+',
        ]

        for pattern in question_patterns:
            reformulated = re.sub(pattern, '', reformulated)

        # Remove trailing question marks
        reformulated = reformulated.rstrip('?').strip()

        # Remove common filler words
        filler_words = [
            r'\b(the|a|an)\b',
            r'\b(in|on|at|to|for|of|with|from)\b',
            r'\b(this|that|these|those)\b',
            r'\b(please|thanks|thank you)\b',
        ]

        for filler in filler_words:
            reformulated = re.sub(filler, ' ', reformulated)

        # Clean up multiple spaces
        reformulated = re.sub(r'\s+', ' ', reformulated).strip()

        # If we've made it too short, use original
        if len(reformulated.split()) < 2:
            reformulated = query

        logger.info(
            "rule_based_reformulation",
            original=query,
            reformulated=reformulated
        )

        return reformulated

    async def close(self):
        """Close provider connections."""
        if self.provider:
            await self.provider.close()


# Singleton instance
_reformulator: Optional[QueryReformulator] = None


def get_query_reformulator() -> QueryReformulator:
    """Get or create query reformulator singleton.

    Returns:
        QueryReformulator instance
    """
    global _reformulator
    if _reformulator is None:
        _reformulator = QueryReformulator()
    return _reformulator
