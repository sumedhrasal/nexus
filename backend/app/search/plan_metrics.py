"""Plan metrics tracking and monitoring for adaptive RAG pipeline."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from app.search.plan_schema import ExecutionPlan, ExecutionStrategy
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PlanMetrics:
    """Tracks metrics for execution plans."""

    # Plan distribution counters
    complexity_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    strategy_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Feature usage counters
    decomposition_used: int = 0
    iterative_used: int = 0

    # Confidence tracking
    total_confidence: float = 0.0
    confidence_count: int = 0
    low_confidence_count: int = 0  # confidence < 0.5

    # Fallback tracking
    fallback_count: int = 0

    # Total plans generated
    total_plans: int = 0

    # Timestamp of last reset
    last_reset: datetime = field(default_factory=datetime.utcnow)

    def record_plan(self, plan: ExecutionPlan, is_fallback: bool = False) -> None:
        """Record a plan for metrics tracking.

        Args:
            plan: Execution plan to record
            is_fallback: Whether this is a fallback plan
        """
        self.total_plans += 1

        # Track complexity distribution
        self.complexity_counts[plan.complexity.value] += 1

        # Track strategy distribution
        self.strategy_counts[plan.strategy.value] += 1

        # Track feature usage
        if plan.use_decomposition:
            self.decomposition_used += 1

        if plan.use_iterative_retrieval:
            self.iterative_used += 1

        # Track confidence
        if plan.confidence is not None:
            self.total_confidence += plan.confidence
            self.confidence_count += 1

            if plan.confidence < 0.5:
                self.low_confidence_count += 1

        # Track fallbacks
        if is_fallback:
            self.fallback_count += 1

        # Log plan details
        logger.info(
            "plan_recorded",
            complexity=plan.complexity.value,
            strategy=plan.strategy.value,
            confidence=plan.confidence,
            use_decomposition=plan.use_decomposition,
            use_iterative=plan.use_iterative_retrieval,
            is_fallback=is_fallback
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Dictionary with metrics summary
        """
        avg_confidence = (
            self.total_confidence / self.confidence_count
            if self.confidence_count > 0
            else None
        )

        return {
            "total_plans": self.total_plans,
            "complexity_distribution": dict(self.complexity_counts),
            "strategy_distribution": dict(self.strategy_counts),
            "decomposition_usage_rate": (
                self.decomposition_used / self.total_plans
                if self.total_plans > 0
                else 0.0
            ),
            "iterative_usage_rate": (
                self.iterative_used / self.total_plans
                if self.total_plans > 0
                else 0.0
            ),
            "average_confidence": avg_confidence,
            "low_confidence_rate": (
                self.low_confidence_count / self.confidence_count
                if self.confidence_count > 0
                else 0.0
            ),
            "fallback_rate": (
                self.fallback_count / self.total_plans
                if self.total_plans > 0
                else 0.0
            ),
            "last_reset": self.last_reset.isoformat()
        }

    def reset(self) -> None:
        """Reset all metrics counters."""
        self.complexity_counts.clear()
        self.strategy_counts.clear()
        self.decomposition_used = 0
        self.iterative_used = 0
        self.total_confidence = 0.0
        self.confidence_count = 0
        self.low_confidence_count = 0
        self.fallback_count = 0
        self.total_plans = 0
        self.last_reset = datetime.utcnow()

        logger.info("plan_metrics_reset")


# Global metrics instance
_metrics: Optional[PlanMetrics] = None


def get_plan_metrics() -> PlanMetrics:
    """Get or create plan metrics singleton.

    Returns:
        PlanMetrics instance
    """
    global _metrics

    if _metrics is None:
        _metrics = PlanMetrics()

    return _metrics


def log_plan_execution_start(plan: ExecutionPlan, query: str) -> None:
    """Log the start of plan execution.

    Args:
        plan: Execution plan being executed
        query: Original query
    """
    logger.info(
        "plan_execution_started",
        query=query[:100],
        query_length=len(query),
        complexity=plan.complexity.value,
        strategy=plan.strategy.value,
        use_decomposition=plan.use_decomposition,
        num_sub_queries=plan.num_sub_queries,
        use_iterative=plan.use_iterative_retrieval,
        max_iterations=plan.max_iterations,
        synthesis_style=plan.synthesis_style.value,
        initial_retrieval_limit=plan.initial_retrieval_limit,
        confidence=plan.confidence
    )


def log_plan_execution_complete(
    plan: ExecutionPlan,
    query: str,
    execution_time_ms: float,
    success: bool,
    error: Optional[str] = None
) -> None:
    """Log the completion of plan execution.

    Args:
        plan: Execution plan that was executed
        query: Original query
        execution_time_ms: Execution time in milliseconds
        success: Whether execution was successful
        error: Optional error message if execution failed
    """
    logger.info(
        "plan_execution_completed",
        query=query[:100],
        complexity=plan.complexity.value,
        strategy=plan.strategy.value,
        execution_time_ms=execution_time_ms,
        success=success,
        error=error
    )


def log_strategy_decision(
    query: str,
    chosen_strategy: ExecutionStrategy,
    reasoning: str,
    alternatives_considered: Optional[list] = None
) -> None:
    """Log strategy selection decision.

    Args:
        query: User query
        chosen_strategy: Strategy that was chosen
        reasoning: LLM's reasoning for the choice
        alternatives_considered: Optional list of alternative strategies considered
    """
    logger.info(
        "strategy_decision",
        query=query[:100],
        chosen_strategy=chosen_strategy.value,
        reasoning=reasoning[:200],
        alternatives=alternatives_considered
    )


def log_decomposition_result(
    query: str,
    sub_queries: list,
    num_generated: int
) -> None:
    """Log query decomposition result.

    Args:
        query: Original query
        sub_queries: Generated sub-queries
        num_generated: Number of sub-queries generated
    """
    logger.info(
        "decomposition_result",
        original_query=query[:100],
        num_sub_queries=num_generated,
        sub_queries=[sq[:50] for sq in sub_queries]  # Log first 50 chars of each
    )


def log_iteration_result(
    iteration: int,
    max_iterations: int,
    should_continue: bool,
    reason: str,
    chunks_retrieved: int
) -> None:
    """Log iterative retrieval iteration result.

    Args:
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        should_continue: Whether to continue iterating
        reason: LLM's reason for the decision
        chunks_retrieved: Number of chunks retrieved in this iteration
    """
    logger.info(
        "iteration_result",
        iteration=iteration,
        max_iterations=max_iterations,
        should_continue=should_continue,
        reason=reason[:200],
        chunks_retrieved=chunks_retrieved
    )
