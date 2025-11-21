"""Text extraction utilities for various document formats."""

import re
from typing import Optional
from bs4 import BeautifulSoup, Tag, NavigableString
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_html(html_content: str, preserve_structure: bool = True) -> str:
    """Extract clean text from HTML content.

    This is a generic approach that works with any HTML structure:
    - Removes scripts, styles, and metadata elements
    - Handles mathematical notation (MathML/LaTeX annotations)
    - Preserves semantic structure with proper spacing
    - Cleans up excessive whitespace

    Args:
        html_content: Raw HTML string
        preserve_structure: If True, preserves paragraph breaks and headings

    Returns:
        Clean text extracted from HTML
    """
    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted elements that add no semantic value
        unwanted_tags = [
            'script', 'style', 'meta', 'link', 'noscript',
            'iframe', 'embed', 'object', 'svg', 'canvas'
        ]
        for tag_name in unwanted_tags:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Handle mathematical notation intelligently
        # Try to extract readable representation or remove entirely
        for math in soup.find_all('math'):
            # Look for LaTeX annotation which is more readable
            annotation = math.find('annotation', attrs={'encoding': 'application/x-tex'})
            if annotation:
                # Replace with LaTeX notation
                math.replace_with(f" ${annotation.get_text().strip()}$ ")
            else:
                # No readable alternative, replace with placeholder
                math.replace_with(" [formula] ")

        # Handle tables - extract cell content with basic structure
        for table in soup.find_all('table'):
            table_text = []
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                    table_text.append(row_text)
            if table_text:
                table.replace_with('\n' + '\n'.join(table_text) + '\n')

        # Get text with structure preservation if requested
        if preserve_structure:
            text = _extract_with_structure(soup)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        # Multiple spaces to single
        text = re.sub(r' +', ' ', text)
        # Multiple newlines to maximum of 2 (paragraph break)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Remove leading/trailing whitespace on each line
        text = '\n'.join(line.strip() for line in text.split('\n'))

        # Remove any remaining HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#?\w+;', ' ', text)

        # Final cleanup
        text = re.sub(r' +', ' ', text)
        text = text.strip()

        logger.debug(
            "html_text_extracted",
            original_length=len(html_content),
            extracted_length=len(text),
            compression_ratio=f"{len(text)/len(html_content)*100:.1f}%"
        )

        return text

    except Exception as e:
        logger.error(
            "html_extraction_failed",
            error=str(e),
            html_preview=html_content[:200]
        )
        # Fallback to simple text extraction if parsing fails
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)


def _extract_with_structure(soup: BeautifulSoup) -> str:
    """Extract text while preserving document structure.

    Maintains:
    - Headings with extra spacing
    - Paragraph breaks
    - List structure
    - Section divisions
    """
    lines = []

    # Process top-level elements
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section', 'article', 'li', 'blockquote']):
        # Skip if this element is inside another element we'll process
        if element.find_parent(['p', 'li', 'blockquote']):
            continue

        text = element.get_text(separator=' ', strip=True)
        if not text:
            continue

        # Add extra spacing for headings
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            lines.append(f"\n{text}\n")
        # Preserve list structure
        elif element.name == 'li':
            lines.append(f"• {text}")
        # Regular paragraphs
        else:
            lines.append(text)

    # If we didn't extract much (maybe unconventional HTML structure),
    # fall back to simple extraction
    result = '\n\n'.join(lines)
    if len(result) < 100:
        result = soup.get_text(separator='\n', strip=True)

    return result


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text.

    Args:
        text: Text to estimate
        chars_per_token: Average characters per token (default: 4 for English)

    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)


def truncate_to_token_limit(text: str, max_tokens: int, chars_per_token: float = 4.0) -> str:
    """Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        chars_per_token: Average characters per token

    Returns:
        Truncated text
    """
    max_chars = int(max_tokens * chars_per_token)
    if len(text) <= max_chars:
        return text

    # Truncate at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.9:  # Within 90% of limit
        truncated = truncated[:last_space]

    return truncated + "..."
