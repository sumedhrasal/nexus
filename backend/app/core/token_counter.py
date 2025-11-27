"""Token counting and text truncation utilities."""

from typing import List
import re


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple heuristic.

    This is a rough approximation. For exact counts, use tiktoken.

    Rule of thumb:
    - 1 token ~= 4 characters in English
    - 1 token ~= 0.75 words in English

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Simple approximation: split on whitespace and punctuation
    # Each word ~= 1.33 tokens on average
    words = re.findall(r'\w+|[^\w\s]', text)
    return int(len(words) * 1.33)


def truncate_to_token_limit(text: str, max_tokens: int, safety_margin: int = 100) -> str:
    """Truncate text to fit within token limit.

    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        safety_margin: Safety margin to stay under limit (default: 100 tokens)

    Returns:
        Truncated text that fits within token limit
    """
    estimated_tokens = estimate_tokens(text)

    if estimated_tokens <= max_tokens:
        return text

    # Calculate target character count based on token estimate
    # Reserve safety margin
    target_tokens = max_tokens - safety_margin
    char_per_token = len(text) / estimated_tokens
    target_chars = int(target_tokens * char_per_token)

    # Truncate to target characters
    truncated = text[:target_chars]

    # Try to truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > target_chars * 0.9:  # If within 90% of target
        truncated = truncated[:last_space]

    return truncated


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
    """Split text into chunks that fit within token limit.

    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    estimated_tokens = estimate_tokens(text)

    if estimated_tokens <= max_tokens:
        return [text]

    chunks = []
    words = text.split()
    current_chunk_words = []

    # Estimate tokens to words ratio for overlap calculation
    words_per_token = len(words) / estimated_tokens if estimated_tokens > 0 else 0.75
    overlap_words = int(overlap_tokens * words_per_token)

    for word in words:
        # Try adding the word
        test_chunk = current_chunk_words + [word]
        test_text = ' '.join(test_chunk)
        test_tokens = estimate_tokens(test_text)

        # If it would exceed the limit and we have words in current chunk
        if test_tokens > max_tokens and current_chunk_words:
            # Save current chunk
            chunks.append(' '.join(current_chunk_words))

            # Start new chunk with overlap
            if overlap_words > 0 and len(current_chunk_words) > overlap_words:
                current_chunk_words = current_chunk_words[-overlap_words:]
            else:
                current_chunk_words = []

            # Add word to new chunk
            current_chunk_words.append(word)
        else:
            # Word fits, add it
            current_chunk_words.append(word)

    # Add final chunk if not empty
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))

    return chunks
