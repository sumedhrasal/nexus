"""LLM-based query planning service.

This module uses an LLM to analyze queries and generate execution plans
for the adaptive RAG pipeline.
"""

import json
from typing import Optional, Dict, Any
from app.core.providers.router import ProviderRouter
from app.search.plan_schema import ExecutionPlan, PlanValidationError
from app.search.plan_validator import get_plan_validator, PlanValidator
from app.search.plan_metrics import get_plan_metrics, log_strategy_decision
from app.search.plan_cache import get_plan_cache
from app.core.logging import get_logger

logger = get_logger(__name__)


class QueryPlanner:
    """Generates execution plans for queries using LLM."""

    def __init__(
        self,
        provider: Optional[ProviderRouter] = None,
        validator: Optional[PlanValidator] = None
    ):
        """Initialize query planner.

        Args:
            provider: LLM provider for plan generation (creates new if not provided)
            validator: Plan validator (uses default if not provided)
        """
        self.provider = provider or ProviderRouter(strategy="cost")  # Use cheapest LLM for planning
        self.validator = validator or get_plan_validator()

    async def create_plan(self, query: str) -> ExecutionPlan:
        """Create execution plan for query.

        Args:
            query: User query

        Returns:
            Validated ExecutionPlan

        Raises:
            Exception: If planning fails completely (caller should handle)
        """
        logger.info(
            "planning_started",
            query=query[:100],  # Log first 100 chars
            query_length=len(query),
            query_word_count=len(query.split())
        )

        try:
            # Check cache first
            cache = await get_plan_cache()
            cached_plan = await cache.get(query)

            if cached_plan:
                logger.info(
                    "planning_completed_from_cache",
                    strategy=cached_plan.strategy.value,
                    complexity=cached_plan.complexity.value,
                    confidence=cached_plan.confidence
                )

                # Still record metrics for cached plans
                metrics = get_plan_metrics()
                metrics.record_plan(cached_plan, is_fallback=False)

                # Log strategy decision
                log_strategy_decision(
                    query=query,
                    chosen_strategy=cached_plan.strategy,
                    reasoning=cached_plan.reasoning
                )

                return cached_plan

            # Step 1: Generate plan using LLM
            plan_data = await self._generate_plan_with_llm(query)

            # Step 2: Validate and sanitize
            plan = self.validator.validate_and_sanitize(plan_data, query)

            logger.info(
                "planning_completed",
                strategy=plan.strategy.value,
                complexity=plan.complexity.value,
                confidence=plan.confidence
            )

            # Cache the new plan
            cache = await get_plan_cache()
            await cache.set(query, plan)

            # Record metrics
            metrics = get_plan_metrics()
            metrics.record_plan(plan, is_fallback=False)

            # Log strategy decision
            log_strategy_decision(
                query=query,
                chosen_strategy=plan.strategy,
                reasoning=plan.reasoning
            )

            return plan

        except Exception as e:
            logger.error(
                "planning_failed",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100],
                exc_info=True
            )

            # Create fallback plan
            fallback_plan = self.validator.create_fallback_plan(
                query,
                reason=f"Planning failed: {str(e)[:100]}"
            )

            # Record fallback metrics
            metrics = get_plan_metrics()
            metrics.record_plan(fallback_plan, is_fallback=True)

            return fallback_plan

    async def _generate_plan_with_llm(self, query: str) -> Dict[str, Any]:
        """Generate plan using LLM.

        Args:
            query: User query

        Returns:
            Raw plan dictionary (not yet validated)

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        system_prompt = self._build_planning_system_prompt()
        user_prompt = self._build_planning_user_prompt(query)

        logger.debug(
            "llm_planning_request",
            system_prompt_length=len(system_prompt),
            user_prompt_length=len(user_prompt)
        )

        # Generate plan with LLM
        response, provider_name = await self.provider.generate(
            prompt=user_prompt,
            system=system_prompt
        )

        logger.debug(
            "llm_planning_response",
            provider=provider_name,
            response_length=len(response)
        )

        # Parse JSON response
        plan_data = self._parse_plan_response(response)

        return plan_data

    def _build_planning_system_prompt(self) -> str:
        """Build system prompt for planning LLM."""
        return """You are an expert query analysis assistant for a RAG (Retrieval-Augmented Generation) system.

Your job is to analyze user queries and create optimal execution plans.

You must return ONLY a valid JSON object with this EXACT structure:
{
  "complexity": "simple|moderate|complex|research",
  "strategy": "direct|decompose_and_synthesize|iterative_deep_dive",
  "reasoning": "brief explanation of your analysis (1-2 sentences)",
  "use_decomposition": true|false,
  "num_sub_queries": null or 2-5 (only if use_decomposition=true),
  "use_iterative_retrieval": true|false,
  "max_iterations": null or 1-5 (only if use_iterative_retrieval=true),
  "synthesis_style": "concise|structured|comprehensive",
  "expected_answer_length": "short|medium|long",
  "initial_retrieval_limit": 5-40 (number of chunks to retrieve),
  "confidence": 0.0-1.0 (your confidence in this plan)
}

GUIDELINES:

**Complexity Classification:**
- simple: Single-fact questions ("What is X?", "When was Y?", "Who created Z?")
- moderate: How/why questions requiring understanding ("How does X work?", "Why is Y important?")
- complex: Multi-faceted questions ("Compare X and Y", "What are the differences between A and B?")
- research: Deep analysis questions ("Comprehensive analysis of...", "Explain all aspects of...")

**Strategy Selection:**
- direct: For simple/moderate queries, single retrieval sufficient
- decompose_and_synthesize: For complex queries needing multiple perspectives
- iterative_deep_dive: For research queries that may need follow-up retrieval

**Decomposition Logic:**
- Use for multi-faceted questions (compare, contrast, multiple aspects)
- NOT for simple factual questions
- 2-3 sub-queries for complex, 3-5 for research

**Iterative Retrieval:**
- Use ONLY for research-level queries where initial context may be insufficient
- NOT for simple or most moderate queries (too slow)
- Max 2-3 iterations for most cases

**Synthesis Style:**
- concise: For simple queries (2-4 sentences)
- structured: For moderate/complex queries (organized sections)
- comprehensive: For research queries (full analysis)

**Important:**
- Be conservative: Prefer simpler strategies when in doubt (avoid over-engineering)
- Consider cost vs. quality: Iterative retrieval is expensive, use sparingly
- Return ONLY valid JSON, no markdown, no extra text"""

    def _build_planning_user_prompt(self, query: str) -> str:
        """Build user prompt for planning LLM.

        Args:
            query: User query

        Returns:
            Formatted user prompt
        """
        query_length = len(query.split())

        return f"""Analyze this query and create an execution plan:

Query: "{query}"

Query Statistics:
- Word count: {query_length}
- Character count: {len(query)}

Create the optimal execution plan in JSON format (as specified in system prompt).

Remember:
- Be conservative (simpler is often better)
- Consider the cost of complex strategies
- Return ONLY the JSON object, nothing else"""

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract plan JSON.

        Args:
            response: Raw LLM response

        Returns:
            Parsed plan dictionary

        Raises:
            ValueError: If response is not valid JSON
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
            plan_data = json.loads(response)

            if not isinstance(plan_data, dict):
                raise ValueError(f"Expected dict, got {type(plan_data)}")

            logger.debug(
                "plan_parsed",
                fields=list(plan_data.keys())
            )

            return plan_data

        except json.JSONDecodeError as e:
            logger.error(
                "plan_json_parse_failed",
                error=str(e),
                response_preview=response[:200],
                exc_info=True
            )
            raise ValueError(f"Failed to parse plan JSON: {e}")

    async def close(self):
        """Close provider connections."""
        if self.provider:
            # ProviderRouter doesn't have close method currently
            # but we keep this for future compatibility
            pass


# Singleton instance
_planner: Optional[QueryPlanner] = None


def get_query_planner(
    provider: Optional[ProviderRouter] = None,
    validator: Optional[PlanValidator] = None
) -> QueryPlanner:
    """Get or create query planner singleton.

    Args:
        provider: Optional provider override
        validator: Optional validator override

    Returns:
        QueryPlanner instance
    """
    global _planner

    if _planner is None:
        _planner = QueryPlanner(provider=provider, validator=validator)

    return _planner
