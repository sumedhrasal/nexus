"""Cross-encoder re-ranking for improved search relevance."""

from typing import List, Dict, Any, Optional
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Re-rank search results using cross-encoder models."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name (default: from settings)
        """
        self.model_name = model_name or settings.reranker_model
        self.model = None
        self._initialized = False

    def _lazy_load_model(self):
        """Lazy load the model only when needed."""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "loading_reranker_model",
                model=self.model_name
            )

            self.model = CrossEncoder(self.model_name, max_length=512)
            self._initialized = True

            logger.info("reranker_model_loaded", model=self.model_name)

        except ImportError:
            logger.error(
                "reranker_import_failed",
                message="sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(
                "reranker_model_load_failed",
                model=self.model_name,
                error=str(e),
                exc_info=True
            )
            raise

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Re-rank search results using cross-encoder.

        Args:
            query: Search query
            results: List of search results with 'content' field
            top_k: Number of top results to return (default: from settings)

        Returns:
            Re-ranked results with updated scores
        """
        if not results:
            return results

        top_k = top_k or settings.reranker_top_k

        # Only rerank if we have more results than top_k
        if len(results) <= top_k:
            logger.debug(
                "reranking_skipped",
                reason="not_enough_results",
                result_count=len(results),
                top_k=top_k
            )
            return results

        try:
            # Lazy load model
            self._lazy_load_model()

            # Prepare query-document pairs
            pairs = [[query, result["content"]] for result in results[:top_k]]

            logger.debug(
                "reranking_started",
                query=query,
                num_candidates=len(pairs)
            )

            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Create reranked results with new scores
            reranked = []
            for i, score in enumerate(scores):
                result = results[i].copy()
                result["original_score"] = result.get("score", 0.0)
                result["reranker_score"] = float(score)
                result["score"] = float(score)  # Replace with reranker score
                reranked.append(result)

            # Sort by reranker score
            reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

            logger.info(
                "reranking_completed",
                num_reranked=len(reranked),
                top_score=reranked[0]["reranker_score"] if reranked else 0.0,
                bottom_score=reranked[-1]["reranker_score"] if reranked else 0.0
            )

            # Add remaining results that weren't reranked
            if len(results) > top_k:
                reranked.extend(results[top_k:])

            return reranked

        except Exception as e:
            logger.error(
                "reranking_failed",
                query=query,
                num_results=len(results),
                error=str(e),
                exc_info=True
            )
            # Fallback: return original results
            return results


# Singleton instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get or create reranker singleton.

    Returns:
        CrossEncoderReranker instance
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
