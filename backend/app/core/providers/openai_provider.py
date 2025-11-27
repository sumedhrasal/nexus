"""OpenAI provider."""

from typing import List, Optional
from openai import AsyncOpenAI

from app.core.providers.base import BaseProvider
from app.config import settings


class OpenAIProvider(BaseProvider):
    """OpenAI provider for embeddings and text generation.

    Uses:
    - text-embedding-3-small (1536d) or text-embedding-3-large (3072d)
    - gpt-4o-mini for text generation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: Optional[str] = None,
        llm_model: Optional[str] = None
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (default: from settings)
            embed_model: Embedding model name (default: from settings)
            llm_model: LLM model name (default: from settings)
        """
        super().__init__(name="openai")
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

        self.client = AsyncOpenAI(api_key=self.api_key)

        # Load model configuration from settings
        self.embed_model = embed_model or settings.openai_embedding_model
        self.llm_model = llm_model or settings.openai_llm_model
        self.dimension = settings.openai_embedding_dimension
        self.context_window = settings.openai_context_window

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using text-embedding-3.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (1536d or 3072d)
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embed_model,
                input=texts,
                encoding_format="float"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}") from e

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text using gpt-4o-mini.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            Generated text
        """
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}") from e

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            1536 (small) or 3072 (large)
        """
        return self.dimension

    def get_cost_per_1k_tokens(self, operation: str = "embed") -> float:
        """Get cost per 1K tokens.

        Args:
            operation: "embed" or "generate"

        Returns:
            Cost in USD per 1K tokens
        """
        if operation == "embed":
            return settings.openai_embed_cost_per_1k
        else:
            return 0.0005  # gpt-4o-mini generation cost

    def get_max_embedding_tokens(self) -> int:
        """Get maximum embedding tokens.

        Returns:
            Maximum tokens for embedding model (text-embedding-3-*: 8191 tokens)
        """
        return settings.openai_context_window

    async def health_check(self) -> bool:
        """Check if OpenAI API is responding.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try embedding a simple test string
            await self.embed(["test"])
            return True
        except Exception:
            return False
