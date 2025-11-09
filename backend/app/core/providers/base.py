"""Base provider interface for AI services."""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseProvider(ABC):
    """Abstract base class for all AI providers (OpenAI, Gemini, Ollama)."""

    def __init__(self, name: str):
        """Initialize provider with name."""
        self.name = name

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text completion from prompt.

        Args:
            prompt: User prompt text
            system: Optional system prompt for context

        Returns:
            Generated text response

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embedding vectors produced by this provider.

        Returns:
            Integer dimension (e.g., 768, 1536, 3072)
        """
        pass

    @abstractmethod
    def get_cost_per_1k_tokens(self, operation: str = "embed") -> float:
        """Get cost per 1K tokens for this provider.

        Args:
            operation: Either "embed" or "generate"

        Returns:
            Cost in USD per 1K tokens (0.0 for Ollama)
        """
        pass

    async def health_check(self) -> bool:
        """Check if provider is healthy and responding.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try embedding a simple test string
            await self.embed(["test"])
            return True
        except Exception:
            return False

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()
