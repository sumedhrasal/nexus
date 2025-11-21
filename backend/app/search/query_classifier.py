"""Query type classification for adaptive search weighting."""

import re
from typing import Tuple
from enum import Enum


class QueryType(Enum):
    """Types of queries for adaptive weighting."""
    QUESTION = "question"  # What/How/Why questions - favor semantic
    KEYWORD = "keyword"    # Short keyword searches - balance both
    TECHNICAL = "technical"  # Technical/specific terms - favor BM25


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
