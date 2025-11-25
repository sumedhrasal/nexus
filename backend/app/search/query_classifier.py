"""Query type classification for adaptive search weighting and hybrid search strategy selection."""

import re
from typing import Tuple, Literal
from enum import Enum
from app.core.logging import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of queries for adaptive weighting."""
    QUESTION = "question"  # What/How/Why questions - favor semantic
    KEYWORD = "keyword"    # Short keyword searches - balance both
    TECHNICAL = "technical"  # Technical/specific terms - favor BM25


# Type alias for search mode classification
SearchMode = Literal["keyword_heavy", "semantic", "hybrid"]


def classify_query(query: str) -> Tuple[QueryType, float, float]:
    """Classify query type and return optimal semantic/BM25 weights.

    Args:
        query: User query string

    Returns:
        Tuple of (query_type, semantic_weight, bm25_weight)

    Examples:
        "what is REFRAG?" -> (QUESTION, 0.8, 0.2)
        "machine learning" -> (KEYWORD, 0.5, 0.5)
        "tensorflow.keras.layers.Dense" -> (TECHNICAL, 0.6, 0.4)
    """
    query_lower = query.lower().strip()

    # Question pattern detection
    question_patterns = [
        r'^(what|how|why|when|where|who|which)\s+(is|are|was|were|do|does|did|can|could|would|should)',
        r'^(explain|describe|tell me|show me)',
        r'\?$',  # Ends with question mark
    ]

    is_question = any(re.search(pattern, query_lower) for pattern in question_patterns)

    if is_question:
        # Question queries: favor semantic understanding
        return QueryType.QUESTION, 0.8, 0.2

    # Technical pattern detection
    technical_patterns = [
        r'[A-Z][a-z]+[A-Z]',  # CamelCase (e.g., "ClassName")
        r'[a-z_]+\.[a-z_]+',  # Dotted notation (e.g., "module.function")
        r'[a-z_]+::[a-z_]+',  # Double colon (e.g., "namespace::class")
        r'\b[A-Z]{2,}\b',     # Acronyms (e.g., "API", "HTTP")
        r'[a-z]+_[a-z]+_[a-z]+',  # Snake_case with multiple underscores
        r'\d+\.\d+\.\d+',     # Version numbers (e.g., "1.2.3")
    ]

    has_technical = any(re.search(pattern, query) for pattern in technical_patterns)

    # Check for programming language keywords
    programming_keywords = {
        'class', 'function', 'method', 'api', 'endpoint', 'parameter',
        'return', 'async', 'await', 'import', 'export', 'interface',
        'type', 'struct', 'enum', 'const', 'var', 'let', 'def'
    }

    query_words = set(query_lower.split())
    has_programming = bool(query_words & programming_keywords)

    if has_technical or has_programming:
        # Technical queries: balance semantic with keyword matching
        return QueryType.TECHNICAL, 0.6, 0.4

    # Short queries (1-3 words) without question markers
    word_count = len(query.split())
    if word_count <= 3:
        # Keyword queries: balance both approaches
        return QueryType.KEYWORD, 0.5, 0.5

    # Default: treat as keyword query
    return QueryType.KEYWORD, 0.5, 0.5


def get_adaptive_weights(query: str) -> Tuple[float, float]:
    """Get adaptive semantic and BM25 weights for a query.

    Args:
        query: User query string

    Returns:
        Tuple of (semantic_weight, bm25_weight)
    """
    _, semantic_weight, bm25_weight = classify_query(query)
    return semantic_weight, bm25_weight


def classify_for_hybrid_search(query: str) -> SearchMode:
    """Classify query for hybrid search strategy selection (Phase 1).

    Uses enhanced pattern matching to determine if query needs:
    - keyword_heavy: BM25 sparse vectors (exact matching critical)
    - semantic: Dense vectors only (conceptual understanding)
    - hybrid: Both strategies combined

    Args:
        query: User search query

    Returns:
        SearchMode: "keyword_heavy", "semantic", or "hybrid"
    """
    query_lower = query.lower()

    # Extended keyword patterns for exact matching scenarios
    keyword_patterns = [
        r'#\d+',                        # Issue numbers: #12345
        r'\b[A-Z][a-z]+\.[a-z]+',       # File names: Auth.py
        r'\bclass\s+\w+',               # Code: "class DatabaseConnection"
        r'\bdef\s+\w+',                 # Code: "def authenticate"
        r'\bfunction\s+\w+',            # Code: "function handleRequest"
        r':\d+',                        # Line numbers: ":47"
        r'@\w+',                        # Decorators: @override
        r'\b[A-Z_]{3,}\b',              # Constants: API_KEY
        r'"[^"]{3,}"',                  # Quoted: "exact phrase"
        r'\b\w+\.\w+\.\w+',             # Dotted paths: config.database.url
        r'\b\w+\.\w+\b',                # Simple dotted: UserService.authenticate
        r'\b[a-z]+_[a-z]+\b',           # Snake case: user_service
        r'\b[A-Z][a-z]+[A-Z]\w*\b',     # CamelCase: UserService
        r'\berror:\s*\w+',              # Errors: "error: connection refused"
        r'\b\w+Error\b',                # Error classes: ValueError
        r'\b\w+Exception\b',            # Exceptions: RuntimeException
    ]

    # Semantic indicators
    semantic_indicators = [
        'how', 'why', 'what', 'when', 'where', 'who', 'which',
        'explain', 'describe', 'tell', 'show', 'understand', 'learn',
        'compare', 'difference', 'similar', 'contrast', 'versus', 'vs',
        'overview', 'summary', 'introduction', 'basics', 'fundamentals',
    ]

    # Count matches
    keyword_score = sum(1 for pattern in keyword_patterns if re.search(pattern, query))
    semantic_score = sum(1 for indicator in semantic_indicators if re.search(rf'\b{indicator}\b', query_lower))

    # Classification logic
    if keyword_score >= 2:
        classification = "keyword_heavy"
    elif semantic_score >= 1 and keyword_score == 0:
        classification = "semantic"
    else:
        classification = "hybrid"

    # Detailed classification logging
    logger.info(
        "🔍 QUERY_CLASSIFICATION",
        query=query[:120],
        classification=classification,
        keyword_patterns_found=keyword_score,
        semantic_indicators_found=semantic_score,
        decision_reason=f"keyword={keyword_score}, semantic={semantic_score}"
    )

    if keyword_score > 0:
        logger.debug(f"   → Detected keyword patterns (count: {keyword_score})")
    if semantic_score > 0:
        logger.debug(f"   → Detected semantic indicators (count: {semantic_score})")

    return classification


def classify_with_override(
    query: str,
    user_override: Literal["semantic", "keyword", "hybrid", "auto"] = "auto"
) -> SearchMode:
    """Classify query with optional user override.

    Args:
        query: User search query
        user_override: User-specified search mode or "auto" for classifier decision

    Returns:
        SearchMode to use for search
    """
    if user_override == "auto":
        return classify_for_hybrid_search(query)
    elif user_override == "semantic":
        return "semantic"
    elif user_override == "keyword":
        return "keyword_heavy"
    else:  # "hybrid"
        return "hybrid"
