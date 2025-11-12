"""Search quality metrics calculation."""

from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
import uuid

from app.models.database import SearchAnalytics, SearchFeedback, Collection


class QualityMetricsService:
    """Service for calculating search quality metrics."""

    @staticmethod
    async def calculate_metrics(
        db: AsyncSession,
        collection_id: uuid.UUID,
        period_days: int = 30
    ) -> dict:
        """Calculate quality metrics for a collection.

        Args:
            db: Database session
            collection_id: Collection UUID
            period_days: Number of days to analyze (default: 30)

        Returns:
            Dictionary with quality metrics
        """
        # Calculate time period
        period_end = datetime.now()
        period_start = period_end - timedelta(days=period_days)

        # Get total searches in period
        total_searches_result = await db.execute(
            select(func.count(SearchAnalytics.id))
            .where(
                and_(
                    SearchAnalytics.collection_id == collection_id,
                    SearchAnalytics.created_at >= period_start,
                    SearchAnalytics.created_at <= period_end
                )
            )
        )
        total_searches = total_searches_result.scalar() or 0

        # Get searches with results
        searches_with_results_result = await db.execute(
            select(func.count(SearchAnalytics.id))
            .where(
                and_(
                    SearchAnalytics.collection_id == collection_id,
                    SearchAnalytics.created_at >= period_start,
                    SearchAnalytics.created_at <= period_end,
                    SearchAnalytics.result_count > 0
                )
            )
        )
        searches_with_results = searches_with_results_result.scalar() or 0

        # Get total feedback
        total_feedback_result = await db.execute(
            select(func.count(SearchFeedback.id))
            .join(SearchAnalytics, SearchFeedback.search_analytics_id == SearchAnalytics.id)
            .where(
                and_(
                    SearchAnalytics.collection_id == collection_id,
                    SearchAnalytics.created_at >= period_start,
                    SearchAnalytics.created_at <= period_end
                )
            )
        )
        total_feedback = total_feedback_result.scalar() or 0

        # Get searches with clicks (at least one feedback)
        searches_with_clicks_result = await db.execute(
            select(func.count(func.distinct(SearchAnalytics.id)))
            .join(SearchFeedback, SearchFeedback.search_analytics_id == SearchAnalytics.id)
            .where(
                and_(
                    SearchAnalytics.collection_id == collection_id,
                    SearchAnalytics.created_at >= period_start,
                    SearchAnalytics.created_at <= period_end
                )
            )
        )
        searches_with_clicks = searches_with_clicks_result.scalar() or 0

        # Calculate Precision@k and MRR
        precision_metrics = await QualityMetricsService._calculate_precision_and_mrr(
            db, collection_id, period_start, period_end
        )

        # Calculate coverage
        coverage = (searches_with_results / total_searches * 100) if total_searches > 0 else 0.0

        # Calculate click-through rate
        ctr = (searches_with_clicks / total_searches * 100) if total_searches > 0 else 0.0

        return {
            "collection_id": collection_id,
            "total_searches": total_searches,
            "total_feedback": total_feedback,
            "precision_at_1": precision_metrics["precision_at_1"],
            "precision_at_3": precision_metrics["precision_at_3"],
            "precision_at_5": precision_metrics["precision_at_5"],
            "precision_at_10": precision_metrics["precision_at_10"],
            "mrr": precision_metrics["mrr"],
            "avg_first_click_position": precision_metrics["avg_first_click_position"],
            "searches_with_results": searches_with_results,
            "searches_with_clicks": searches_with_clicks,
            "coverage": round(coverage, 2),
            "click_through_rate": round(ctr, 2),
            "period_start": period_start,
            "period_end": period_end
        }

    @staticmethod
    async def _calculate_precision_and_mrr(
        db: AsyncSession,
        collection_id: uuid.UUID,
        period_start: datetime,
        period_end: datetime
    ) -> dict:
        """Calculate precision@k and MRR metrics.

        Args:
            db: Database session
            collection_id: Collection UUID
            period_start: Start of time period
            period_end: End of time period

        Returns:
            Dictionary with precision and MRR metrics
        """
        # Get all feedback with their positions
        feedback_result = await db.execute(
            select(
                SearchFeedback.search_analytics_id,
                SearchFeedback.result_position,
                SearchFeedback.feedback_type
            )
            .join(SearchAnalytics, SearchFeedback.search_analytics_id == SearchAnalytics.id)
            .where(
                and_(
                    SearchAnalytics.collection_id == collection_id,
                    SearchAnalytics.created_at >= period_start,
                    SearchAnalytics.created_at <= period_end
                )
            )
            .order_by(SearchFeedback.search_analytics_id, SearchFeedback.result_position)
        )
        feedbacks = feedback_result.all()

        if not feedbacks:
            return {
                "precision_at_1": 0.0,
                "precision_at_3": 0.0,
                "precision_at_5": 0.0,
                "precision_at_10": 0.0,
                "mrr": 0.0,
                "avg_first_click_position": 0.0
            }

        # Group feedback by search
        search_feedback_map = {}
        for search_id, position, feedback_type in feedbacks:
            if search_id not in search_feedback_map:
                search_feedback_map[search_id] = []
            search_feedback_map[search_id].append((position, feedback_type))

        # Calculate metrics
        precision_at_1_sum = 0
        precision_at_3_sum = 0
        precision_at_5_sum = 0
        precision_at_10_sum = 0
        reciprocal_ranks = []
        first_click_positions = []

        for search_id, feedback_list in search_feedback_map.items():
            # Sort by position
            feedback_list.sort(key=lambda x: x[0])

            # Get clicked positions (any feedback type considered relevant)
            clicked_positions = [pos for pos, _ in feedback_list]

            # Precision@k: proportion of top-k results that were clicked
            if len(clicked_positions) > 0:
                # Precision@1
                if 0 in clicked_positions:
                    precision_at_1_sum += 1.0

                # Precision@3
                clicks_in_top_3 = sum(1 for pos in clicked_positions if pos < 3)
                precision_at_3_sum += clicks_in_top_3 / min(3, len(clicked_positions))

                # Precision@5
                clicks_in_top_5 = sum(1 for pos in clicked_positions if pos < 5)
                precision_at_5_sum += clicks_in_top_5 / min(5, len(clicked_positions))

                # Precision@10
                clicks_in_top_10 = sum(1 for pos in clicked_positions if pos < 10)
                precision_at_10_sum += clicks_in_top_10 / min(10, len(clicked_positions))

                # MRR: reciprocal of rank of first clicked result
                first_clicked_position = min(clicked_positions)
                reciprocal_ranks.append(1.0 / (first_clicked_position + 1))  # +1 because positions are 0-indexed
                first_click_positions.append(first_clicked_position)

        num_searches_with_feedback = len(search_feedback_map)

        return {
            "precision_at_1": round(precision_at_1_sum / num_searches_with_feedback, 4) if num_searches_with_feedback > 0 else 0.0,
            "precision_at_3": round(precision_at_3_sum / num_searches_with_feedback, 4) if num_searches_with_feedback > 0 else 0.0,
            "precision_at_5": round(precision_at_5_sum / num_searches_with_feedback, 4) if num_searches_with_feedback > 0 else 0.0,
            "precision_at_10": round(precision_at_10_sum / num_searches_with_feedback, 4) if num_searches_with_feedback > 0 else 0.0,
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            "avg_first_click_position": round(sum(first_click_positions) / len(first_click_positions), 2) if first_click_positions else 0.0
        }
