"""Response synthesis service for RAG (Retrieval-Augmented Generation)."""

from typing import List, Optional, Tuple
from app.core.providers.router import ProviderRouter
from app.models.schemas import SearchResult
from app.search.plan_schema import ExecutionPlan, SynthesisStyle


class ResponseSynthesisService:
    """Service for synthesizing answers from search results using LLM."""

    def __init__(self, provider: Optional[ProviderRouter] = None):
        """Initialize response synthesis service.

        Args:
            provider: ProviderRouter instance (creates new one if not provided)
        """
        self.provider = provider or ProviderRouter()

    async def synthesize_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        max_context_chunks: int = 5,
        plan: Optional[ExecutionPlan] = None
    ) -> Tuple[str, int]:
        """Synthesize answer from search results with plan-aware prompting.

        Args:
            query: Original user query
            search_results: List of search results to synthesize from
            max_context_chunks: Maximum number of chunks to use as context
            plan: Optional execution plan for style guidance

        Returns:
            Tuple of (synthesized_answer, estimated_tokens_used)
        """
        if not search_results:
            return "No relevant information found to answer this query.", 0

        # Prepare context from top results
        context_chunks = search_results[:max_context_chunks]
        context_text = self._format_context(context_chunks)

        # Count approximate tokens (rough estimate: 1 token ≈ 4 characters)
        estimated_tokens = len(context_text) // 4 + len(query) // 4 + 200  # +200 for prompt overhead

        # Build plan-aware system prompt
        system_prompt = self._build_synthesis_prompt(plan)

        user_prompt = f"""Context:
{context_text}

Question: {query}

Please provide a concise answer based on the context above."""

        try:
            # Generate synthesized answer using ProviderRouter (tries providers in order)
            answer, provider_name = await self.provider.generate(
                prompt=user_prompt,
                system=system_prompt
            )

            # Clean up the answer
            answer = answer.strip()

            # Update token estimate with response
            estimated_tokens += len(answer) // 4

            return answer, estimated_tokens

        except Exception as e:
            print(f"Response synthesis failed: {e}")
            # Fallback: return a basic summary
            fallback = f"Found {len(search_results)} relevant results. Here's the top match: {search_results[0].content[:200]}..."
            return fallback, estimated_tokens

    def _format_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into hierarchical context for LLM.

        Uses parent-child chunking:
        - High relevance: Full parent content
        - Medium relevance: Expanded context
        - Low relevance: Just child chunk

        Args:
            search_results: List of search results

        Returns:
            Formatted hierarchical context string
        """
        # Organize results by relevance tier
        high_relevance = []  # Top 40% - use parent content
        medium_relevance = []  # Mid 40% - use expanded content
        low_relevance = []  # Bottom 20% - use child chunk only

        total = len(search_results)
        high_threshold = int(total * 0.4)
        medium_threshold = int(total * 0.8)

        for i, result in enumerate(search_results):
            if i < high_threshold:
                high_relevance.append(result)
            elif i < medium_threshold:
                medium_relevance.append(result)
            else:
                low_relevance.append(result)

        context_parts = []

        # High relevance section - use full chunk content
        if high_relevance:
            context_parts.append("=== HIGHLY RELEVANT CONTEXT ===\n")
            for i, result in enumerate(high_relevance, 1):
                title_text = f"[{result.title}]\n" if result.title else ""

                # NOTE: Parent-child chunking disabled, using simple content
                content = result.content

                context_parts.append(
                    f"{i}. {title_text}{content}\n(Relevance: {result.score:.2f})"
                )

        # Medium relevance section
        if medium_relevance:
            context_parts.append("\n=== MODERATELY RELEVANT CONTEXT ===\n")
            for i, result in enumerate(medium_relevance, 1):
                title_text = f"[{result.title}]\n" if result.title else ""

                # NOTE: Parent-child chunking disabled, using simple content
                content = result.content

                context_parts.append(
                    f"{i}. {title_text}{content}\n(Relevance: {result.score:.2f})"
                )

        # Low relevance section - child chunks only (brief)
        if low_relevance:
            context_parts.append("\n=== BACKGROUND CONTEXT ===\n")
            for i, result in enumerate(low_relevance, 1):
                # Just child chunk for background
                context_parts.append(
                    f"{i}. {result.content[:200]}... (Relevance: {result.score:.2f})"
                )

        return "\n\n".join(context_parts)

    def _build_synthesis_prompt(self, plan: Optional[ExecutionPlan]) -> str:
        """Build synthesis system prompt based on execution plan.

        Args:
            plan: Optional execution plan

        Returns:
            System prompt tailored to plan's synthesis style
        """
        base_rules = """You are a helpful assistant that answers questions based on the provided context.

Core Rules:
1. Answer the question using ONLY the information from the context provided
2. If the context doesn't contain enough information, say so honestly
3. Cite relevant information naturally without using citation markers like [1] or [2]
4. Do not make up information or use knowledge outside the provided context"""

        if not plan:
            # Default: concise style
            return base_rules + "\n5. Be concise and direct - aim for 2-4 sentences maximum"

        # Add style-specific guidance
        if plan.synthesis_style == SynthesisStyle.CONCISE:
            style_guidance = """
5. Be extremely concise and direct - aim for 2-4 sentences maximum
6. Focus on the most essential information only
7. Use simple, clear language"""

        elif plan.synthesis_style == SynthesisStyle.STRUCTURED:
            style_guidance = """
5. Organize your answer with clear structure (use bullet points or sections when appropriate)
6. Aim for 1-2 well-organized paragraphs
7. Include key details and supporting information
8. Use headings or bullets to improve readability"""

        elif plan.synthesis_style == SynthesisStyle.COMPREHENSIVE:
            style_guidance = """
5. Provide a thorough, comprehensive answer covering all relevant aspects
6. Include context, details, and examples from the provided information
7. Organize information logically with multiple paragraphs if needed
8. Explain concepts clearly and completely
9. Address different facets of the question systematically"""

        else:
            style_guidance = "\n5. Be concise and direct - aim for 2-4 sentences maximum"

        # Add complexity-aware guidance
        if plan.complexity:
            if plan.complexity.value == "simple":
                complexity_note = "\n\nNote: This is a simple factual question - provide a direct, straightforward answer."
            elif plan.complexity.value == "moderate":
                complexity_note = "\n\nNote: This question requires some explanation - provide clear reasoning and context."
            elif plan.complexity.value == "complex":
                complexity_note = "\n\nNote: This is a multi-faceted question - address different aspects systematically."
            elif plan.complexity.value == "research":
                complexity_note = "\n\nNote: This requires deep analysis - provide comprehensive coverage with full context."
            else:
                complexity_note = ""
        else:
            complexity_note = ""

        return base_rules + style_guidance + complexity_note

    async def close(self):
        """Close provider connections."""
        if self.provider:
            await self.provider.close()


# Singleton instance
_synthesis_service: Optional[ResponseSynthesisService] = None


def get_synthesis_service() -> ResponseSynthesisService:
    """Get or create response synthesis service singleton.

    Returns:
        ResponseSynthesisService instance
    """
    global _synthesis_service
    if _synthesis_service is None:
        _synthesis_service = ResponseSynthesisService()
    return _synthesis_service
