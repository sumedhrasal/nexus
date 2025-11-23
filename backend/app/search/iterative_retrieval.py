"""Iterative retrieval engine with self-assessment.

This module implements adaptive retrieval where an LLM assesses whether
retrieved context is sufficient and decides if more retrieval is needed.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from app.storage.qdrant import QdrantStorage
from app.core.providers.router import ProviderRouter
from app.search.plan_metrics import log_iteration_result
from app.core.logging import get_logger

logger = get_logger(__name__)


class IterativeRetrievalEngine:
    """Handles iterative retrieval with LLM self-assessment."""

    def __init__(
        self,
        qdrant: QdrantStorage,
        provider: ProviderRouter
    ):
        """Initialize iterative retrieval engine.

        Args:
            qdrant: Qdrant storage client for vector search
            provider: Provider router for embeddings and LLM
        """
        self.qdrant = qdrant
        self.provider = provider

    async def retrieve_iteratively(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        max_iterations: int = 3,
        initial_limit: int = 10,
        additional_limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Perform iterative retrieval with self-assessment.

        Strategy:
        1. Retrieve initial set of chunks
        2. Ask LLM if context is sufficient to answer the query
        3. If insufficient, generate follow-up query and retrieve more
        4. Repeat until sufficient or max iterations reached

        Args:
            query: User query
            collection_id: Collection to search in
            vector_dimension: Expected vector dimension
            max_iterations: Maximum number of retrieval iterations
            initial_limit: Number of chunks in first retrieval
            additional_limit: Number of additional chunks per iteration
            filters: Optional filters

        Returns:
            Tuple of (all retrieved chunks, number of iterations performed)
        """
        logger.info(
            "iterative_retrieval_started",
            query=query[:100],
            collection_id=collection_id,
            max_iterations=max_iterations,
            initial_limit=initial_limit
        )

        all_chunks: List[Dict[str, Any]] = []
        iterations_performed = 0

        # Iteration 1: Initial retrieval
        try:
            chunks = await self._retrieve_chunks(
                query=query,
                collection_id=collection_id,
                vector_dimension=vector_dimension,
                limit=initial_limit,
                filters=filters
            )
            all_chunks.extend(chunks)
            iterations_performed = 1

            logger.debug(
                "initial_retrieval_completed",
                chunks_retrieved=len(chunks)
            )

        except Exception as e:
            logger.error(
                "initial_retrieval_failed",
                error=str(e),
                exc_info=True
            )
            raise

        # Iterative assessment and retrieval
        for iteration in range(2, max_iterations + 1):
            try:
                # Assess if we have sufficient context
                should_continue, reason, follow_up_query = await self._assess_sufficiency(
                    query=query,
                    retrieved_chunks=all_chunks,
                    iteration=iteration,
                    max_iterations=max_iterations
                )

                # Log iteration result
                log_iteration_result(
                    iteration=iteration,
                    max_iterations=max_iterations,
                    should_continue=should_continue,
                    reason=reason,
                    chunks_retrieved=len(all_chunks)
                )

                if not should_continue:
                    logger.info(
                        "iterative_retrieval_sufficient",
                        iteration=iteration,
                        total_chunks=len(all_chunks),
                        reason=reason[:100]
                    )
                    break

                # Retrieve additional chunks with follow-up query
                logger.debug(
                    "additional_retrieval_started",
                    iteration=iteration,
                    follow_up_query=follow_up_query[:100]
                )

                additional_chunks = await self._retrieve_chunks(
                    query=follow_up_query,
                    collection_id=collection_id,
                    vector_dimension=vector_dimension,
                    limit=additional_limit,
                    filters=filters,
                    exclude_ids=[chunk["entity_id"] for chunk in all_chunks]
                )

                # Deduplicate (by entity_id)
                existing_ids = {chunk["entity_id"] for chunk in all_chunks}
                new_chunks = [
                    chunk for chunk in additional_chunks
                    if chunk["entity_id"] not in existing_ids
                ]

                all_chunks.extend(new_chunks)
                iterations_performed = iteration

                logger.debug(
                    "additional_retrieval_completed",
                    iteration=iteration,
                    new_chunks=len(new_chunks),
                    total_chunks=len(all_chunks)
                )

            except Exception as e:
                logger.error(
                    "iteration_failed",
                    iteration=iteration,
                    error=str(e),
                    exc_info=True
                )
                # Stop iterating on error, use what we have
                break

        logger.info(
            "iterative_retrieval_completed",
            iterations_performed=iterations_performed,
            total_chunks=len(all_chunks),
            max_iterations=max_iterations
        )

        return all_chunks, iterations_performed

    async def _retrieve_chunks(
        self,
        query: str,
        collection_id: str,
        vector_dimension: int,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks for a query.

        Args:
            query: Query text
            collection_id: Collection to search
            vector_dimension: Expected vector dimension
            limit: Number of chunks to retrieve
            filters: Optional filters
            exclude_ids: Optional list of entity IDs to exclude

        Returns:
            List of retrieved chunks
        """
        # Embed query
        query_embeddings, provider_name = await self.provider.embed(
            [query],
            required_dimension=vector_dimension
        )
        query_embedding = query_embeddings[0]

        # Search
        results = await self.qdrant.search(
            collection_id=collection_id,
            query_vector=query_embedding,
            limit=limit,
            filters=filters
        )

        # Filter out excluded IDs
        if exclude_ids:
            results = [r for r in results if r["entity_id"] not in exclude_ids]

        logger.debug(
            "chunks_retrieved",
            query=query[:50],
            chunks_found=len(results),
            provider=provider_name
        )

        return results

    async def _assess_sufficiency(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        iteration: int,
        max_iterations: int
    ) -> Tuple[bool, str, str]:
        """Assess if retrieved context is sufficient using LLM.

        Args:
            query: Original user query
            retrieved_chunks: All chunks retrieved so far
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed

        Returns:
            Tuple of (should_continue, reason, follow_up_query)
        """
        system_prompt = self._build_assessment_system_prompt()
        user_prompt = self._build_assessment_user_prompt(
            query=query,
            chunks=retrieved_chunks,
            iteration=iteration,
            max_iterations=max_iterations
        )

        logger.debug(
            "llm_assessment_request",
            iteration=iteration,
            num_chunks=len(retrieved_chunks)
        )

        # Get LLM assessment
        response, provider_name = await self.provider.generate(
            prompt=user_prompt,
            system=system_prompt
        )

        logger.debug(
            "llm_assessment_response",
            provider=provider_name,
            response_length=len(response)
        )

        # Parse assessment
        should_continue, reason, follow_up_query = self._parse_assessment_response(response)

        return should_continue, reason, follow_up_query

    def _build_assessment_system_prompt(self) -> str:
        """Build system prompt for sufficiency assessment."""
        return """You are a context sufficiency evaluator for a search system.

Your job is to determine if the retrieved context is sufficient to answer the user's query comprehensively.

You must return ONLY a valid JSON object with this EXACT structure:
{
  "sufficient": true|false,
  "reasoning": "explanation of why context is sufficient or insufficient (1-2 sentences)",
  "follow_up_query": "a refined search query to get missing information (only if insufficient, otherwise null)"
}

EVALUATION CRITERIA:

**Context is SUFFICIENT when:**
- Retrieved chunks directly answer the query
- All aspects of the query are covered
- Information is specific and detailed enough
- No major gaps or missing information

**Context is INSUFFICIENT when:**
- Query asks for multiple aspects but only some are covered
- Information is too vague or high-level
- Retrieved chunks are tangentially related but don't answer directly
- Important details or examples are missing

**Follow-up Query Guidelines (when insufficient):**
- Focus on the specific missing information
- Be more specific than original query
- Use different keywords or phrasing
- Target gaps in current context

**Important:**
- Be conservative: prefer sufficient when context is reasonable
- Consider diminishing returns: after 2-3 iterations, usually sufficient
- Return ONLY valid JSON, no markdown, no extra text"""

    def _build_assessment_user_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        iteration: int,
        max_iterations: int
    ) -> str:
        """Build user prompt for sufficiency assessment.

        Args:
            query: User query
            chunks: Retrieved chunks
            iteration: Current iteration
            max_iterations: Max iterations

        Returns:
            Formatted prompt
        """
        # Combine chunk content (limit length)
        max_content_length = 3000  # Prevent prompt from being too long
        context_parts = []
        current_length = 0

        for idx, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            if current_length + len(content) > max_content_length:
                # Truncate last chunk
                remaining = max_content_length - current_length
                if remaining > 100:
                    context_parts.append(f"[Chunk {idx+1}] {content[:remaining]}... [truncated]")
                break
            context_parts.append(f"[Chunk {idx+1}] {content}")
            current_length += len(content)

        context = "\n\n".join(context_parts)

        return f"""Evaluate if the retrieved context is sufficient to answer the user's query.

Original Query: "{query}"

Retrieved Context ({len(chunks)} chunks, iteration {iteration}/{max_iterations}):
{context}

Assess:
1. Does this context adequately answer the query?
2. Are there significant gaps or missing information?
3. If insufficient, what specific information is missing?

Return the JSON assessment as specified in the system prompt."""

    def _parse_assessment_response(
        self,
        response: str
    ) -> Tuple[bool, str, str]:
        """Parse LLM assessment response.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (should_continue, reason, follow_up_query)
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

            sufficient = data.get("sufficient", True)  # Default to sufficient on parse error
            reasoning = data.get("reasoning", "Context appears sufficient")
            follow_up_query = data.get("follow_up_query", "")

            # Invert: should_continue = NOT sufficient
            should_continue = not sufficient

            logger.debug(
                "assessment_parsed",
                sufficient=sufficient,
                should_continue=should_continue,
                reasoning=reasoning[:100]
            )

            return should_continue, reasoning, follow_up_query or ""

        except json.JSONDecodeError as e:
            logger.error(
                "assessment_json_parse_failed",
                error=str(e),
                response_preview=response[:200],
                exc_info=True
            )
            # Default to sufficient (stop iterating) on parse error
            return False, "Failed to parse assessment, assuming sufficient", ""


async def retrieve_iteratively(
    query: str,
    collection_id: str,
    vector_dimension: int,
    qdrant: QdrantStorage,
    provider: ProviderRouter,
    max_iterations: int = 3,
    initial_limit: int = 10,
    additional_limit: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """Convenience function for iterative retrieval.

    Args:
        query: User query
        collection_id: Collection to search
        vector_dimension: Expected vector dimension
        qdrant: Qdrant storage client
        provider: Provider router
        max_iterations: Max iterations
        initial_limit: Initial retrieval limit
        additional_limit: Additional chunks per iteration
        filters: Optional filters

    Returns:
        Tuple of (all chunks, iterations performed)
    """
    engine = IterativeRetrievalEngine(qdrant=qdrant, provider=provider)
    return await engine.retrieve_iteratively(
        query=query,
        collection_id=collection_id,
        vector_dimension=vector_dimension,
        max_iterations=max_iterations,
        initial_limit=initial_limit,
        additional_limit=additional_limit,
        filters=filters
    )
