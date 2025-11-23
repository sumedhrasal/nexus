"""Plan validation with safety limits and constraints.

This module provides validation logic for execution plans, ensuring they
meet safety constraints and business rules before execution.
"""

from typing import Dict, Any, Optional
from app.search.plan_schema import (
    ExecutionPlan,
    ExecutionStrategy,
    QueryComplexity,
    SynthesisStyle,
    PlanValidationError
)
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PlanValidator:
    """Validates and sanitizes execution plans with safety limits."""

    def __init__(
        self,
        max_sub_queries: int = 5,
        max_iterations: int = 5,
        max_initial_retrieval: int = 40,
        enable_iterative: bool = True,
        enable_decomposition: bool = True
    ):
        """Initialize plan validator.

        Args:
            max_sub_queries: Maximum allowed sub-queries for decomposition
            max_iterations: Maximum allowed iterations for iterative retrieval
            max_initial_retrieval: Maximum chunks in initial retrieval
            enable_iterative: Whether iterative retrieval is allowed
            enable_decomposition: Whether query decomposition is allowed
        """
        self.max_sub_queries = max_sub_queries
        self.max_iterations = max_iterations
        self.max_initial_retrieval = max_initial_retrieval
        self.enable_iterative = enable_iterative
        self.enable_decomposition = enable_decomposition

    def validate_and_sanitize(
        self,
        plan_data: Dict[str, Any],
        original_query: str
    ) -> ExecutionPlan:
        """Validate plan data and create sanitized ExecutionPlan.

        Args:
            plan_data: Raw plan dictionary from LLM
            original_query: Original user query (for context)

        Returns:
            Validated and sanitized ExecutionPlan

        Raises:
            PlanValidationError: If plan fails validation
        """
        logger.debug(
            "validating_plan",
            plan_data=plan_data,
            query_length=len(original_query)
        )

        # Step 1: Check required fields
        self._validate_required_fields(plan_data)

        # Step 2: Apply safety limits
        sanitized_data = self._apply_safety_limits(plan_data)

        # Step 3: Check feature flags
        sanitized_data = self._check_feature_flags(sanitized_data)

        # Step 4: Create and validate Pydantic model
        try:
            plan = ExecutionPlan(**sanitized_data)
        except Exception as e:
            logger.error(
                "plan_schema_validation_failed",
                error=str(e),
                plan_data=sanitized_data,
                exc_info=True
            )
            raise PlanValidationError(f"Plan schema validation failed: {e}")

        # Step 5: Business logic validation
        self._validate_business_rules(plan, original_query)

        logger.info(
            "plan_validated",
            strategy=plan.strategy.value,
            complexity=plan.complexity.value,
            use_decomposition=plan.use_decomposition,
            use_iterative=plan.use_iterative_retrieval
        )

        return plan

    def _validate_required_fields(self, plan_data: Dict[str, Any]) -> None:
        """Check that required fields are present."""
        required_fields = ["complexity", "strategy", "reasoning"]

        missing = [field for field in required_fields if field not in plan_data]

        if missing:
            raise PlanValidationError(
                f"Missing required fields in plan: {missing}"
            )

    def _apply_safety_limits(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hard safety limits to plan parameters."""
        sanitized = plan_data.copy()

        # Limit sub-queries
        if "num_sub_queries" in sanitized and sanitized["num_sub_queries"] is not None:
            original = sanitized["num_sub_queries"]
            sanitized["num_sub_queries"] = min(
                max(2, sanitized["num_sub_queries"]),  # Min 2
                self.max_sub_queries  # Max from config
            )
            if sanitized["num_sub_queries"] != original:
                logger.warning(
                    "num_sub_queries_clamped",
                    original=original,
                    clamped=sanitized["num_sub_queries"],
                    max_allowed=self.max_sub_queries
                )

        # Limit iterations
        if "max_iterations" in sanitized and sanitized["max_iterations"] is not None:
            original = sanitized["max_iterations"]
            sanitized["max_iterations"] = min(
                max(1, sanitized["max_iterations"]),  # Min 1
                self.max_iterations  # Max from config
            )
            if sanitized["max_iterations"] != original:
                logger.warning(
                    "max_iterations_clamped",
                    original=original,
                    clamped=sanitized["max_iterations"],
                    max_allowed=self.max_iterations
                )

        # Limit initial retrieval
        if "initial_retrieval_limit" in sanitized:
            original = sanitized["initial_retrieval_limit"]
            sanitized["initial_retrieval_limit"] = min(
                max(5, sanitized["initial_retrieval_limit"]),  # Min 5
                self.max_initial_retrieval  # Max from config
            )
            if sanitized["initial_retrieval_limit"] != original:
                logger.warning(
                    "initial_retrieval_limit_clamped",
                    original=original,
                    clamped=sanitized["initial_retrieval_limit"],
                    max_allowed=self.max_initial_retrieval
                )

        return sanitized

    def _check_feature_flags(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if requested features are enabled."""
        sanitized = plan_data.copy()

        # Check iterative retrieval
        if sanitized.get("use_iterative_retrieval", False):
            if not self.enable_iterative:
                logger.warning(
                    "iterative_retrieval_disabled",
                    message="Iterative retrieval requested but disabled in config"
                )
                sanitized["use_iterative_retrieval"] = False
                sanitized["max_iterations"] = None

                # Downgrade strategy if necessary
                if sanitized.get("strategy") == "iterative_deep_dive":
                    sanitized["strategy"] = "decompose_and_synthesize" if sanitized.get("use_decomposition") else "direct"
                    logger.warning(
                        "strategy_downgraded",
                        from_strategy="iterative_deep_dive",
                        to_strategy=sanitized["strategy"]
                    )

        # Check query decomposition
        if sanitized.get("use_decomposition", False):
            if not self.enable_decomposition:
                logger.warning(
                    "decomposition_disabled",
                    message="Query decomposition requested but disabled in config"
                )
                sanitized["use_decomposition"] = False
                sanitized["num_sub_queries"] = None

                # Downgrade strategy if necessary
                if sanitized.get("strategy") in ["decompose_and_synthesize", "iterative_deep_dive"]:
                    sanitized["strategy"] = "direct"
                    logger.warning(
                        "strategy_downgraded",
                        from_strategy=plan_data.get("strategy"),
                        to_strategy="direct"
                    )

        return sanitized

    def _validate_business_rules(self, plan: ExecutionPlan, query: str) -> None:
        """Validate business logic rules."""

        # Rule 1: Research strategy should use both decomposition and iteration
        if plan.complexity == QueryComplexity.RESEARCH:
            if plan.strategy == ExecutionStrategy.ITERATIVE:
                if not plan.use_decomposition:
                    logger.warning(
                        "research_without_decomposition",
                        message="Research query using iterative strategy without decomposition (may be suboptimal)"
                    )

        # Rule 2: Simple queries should not use complex strategies
        if plan.complexity == QueryComplexity.SIMPLE:
            if plan.strategy != ExecutionStrategy.DIRECT:
                logger.warning(
                    "simple_query_complex_strategy",
                    complexity="simple",
                    strategy=plan.strategy.value,
                    message="Simple query using complex strategy (may waste resources)"
                )

        # Rule 3: Very short queries probably don't need decomposition
        if len(query.split()) <= 5:  # 5 words or less
            if plan.use_decomposition:
                logger.warning(
                    "short_query_decomposition",
                    query_length=len(query.split()),
                    message="Very short query using decomposition (may be unnecessary)"
                )

        # Rule 4: Confidence check
        if plan.confidence is not None and plan.confidence < 0.5:
            logger.warning(
                "low_confidence_plan",
                confidence=plan.confidence,
                strategy=plan.strategy.value,
                message="LLM has low confidence in this plan"
            )

    def create_fallback_plan(self, query: str, reason: str) -> ExecutionPlan:
        """Create safe fallback plan when LLM planning fails.

        Args:
            query: Original query
            reason: Reason for fallback

        Returns:
            Safe fallback ExecutionPlan (always DIRECT strategy)
        """
        logger.warning(
            "creating_fallback_plan",
            reason=reason,
            query_length=len(query)
        )

        return ExecutionPlan(
            complexity=QueryComplexity.MODERATE,  # Assume moderate
            strategy=ExecutionStrategy.DIRECT,     # Safe default
            reasoning=f"Fallback plan: {reason}",
            use_decomposition=False,
            num_sub_queries=None,
            use_iterative_retrieval=False,
            max_iterations=None,
            synthesis_style=SynthesisStyle.STRUCTURED,
            expected_answer_length="medium",
            initial_retrieval_limit=10,
            confidence=0.5  # Low confidence (fallback)
        )


# Global validator instance
_validator: Optional[PlanValidator] = None


def get_plan_validator(
    max_sub_queries: Optional[int] = None,
    max_iterations: Optional[int] = None,
    max_initial_retrieval: Optional[int] = None,
    enable_iterative: Optional[bool] = None,
    enable_decomposition: Optional[bool] = None
) -> PlanValidator:
    """Get or create plan validator singleton.

    Args:
        max_sub_queries: Override max sub-queries (default: from settings)
        max_iterations: Override max iterations (default: from settings)
        max_initial_retrieval: Override max initial retrieval (default: from settings)
        enable_iterative: Override iterative feature flag (default: from settings)
        enable_decomposition: Override decomposition feature flag (default: from settings)

    Returns:
        PlanValidator instance
    """
    global _validator

    if _validator is None:
        _validator = PlanValidator(
            max_sub_queries=max_sub_queries or getattr(settings, 'max_sub_queries', 5),
            max_iterations=max_iterations or getattr(settings, 'max_rag_iterations', 5),
            max_initial_retrieval=max_initial_retrieval or 40,
            enable_iterative=enable_iterative if enable_iterative is not None else getattr(settings, 'enable_iterative_rag', True),
            enable_decomposition=enable_decomposition if enable_decomposition is not None else getattr(settings, 'enable_query_decomposition', True)
        )

    return _validator
