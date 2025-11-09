"""Google Gemini provider."""

from typing import List, Optional
import google.generativeai as genai

from app.core.providers.base import BaseProvider
from app.config import settings


class GeminiProvider(BaseProvider):
    """Google Gemini provider for embeddings and text generation.

    Uses:
    - text-embedding-004 for embeddings (768 dimensions)
    - gemini-2.0-flash-exp for text generation
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini provider.

        Args:
            api_key: Google API key (default: from settings)
        """
        super().__init__(name="gemini")
        self.api_key = api_key or settings.gemini_api_key

        if not self.api_key or self.api_key == "dummy-key-replace-later":
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)
        self.embed_model_name = "models/text-embedding-004"
        self.llm_model_name = "gemini-2.0-flash-exp"
        self.dimension = 768

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using text-embedding-004.

        Args:
            texts: List of texts to embed

        Returns:
            List of 768-dimensional embedding vectors
        """
        try:
            # Gemini embeddings API
            result = genai.embed_content(
                model=self.embed_model_name,
                content=texts,
                task_type="retrieval_document"
            )

            # Return embeddings list
            if isinstance(result, dict) and "embedding" in result:
                # Single text case
                return [result["embedding"]]
            else:
                # Multiple texts case
                return result["embedding"] if "embedding" in result else result
        except Exception as e:
            raise RuntimeError(f"Gemini embedding failed: {e}") from e

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text using gemini-2.0-flash-exp.

        Args:
            prompt: User prompt
            system: Optional system prompt (prepended to user prompt)

        Returns:
            Generated text
        """
        try:
            model = genai.GenerativeModel(self.llm_model_name)

            # Combine system and user prompts
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            768 (text-embedding-004 dimension)
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
            return settings.gemini_embed_cost_per_1k
        else:
            return 0.0001  # Gemini generation cost

    async def health_check(self) -> bool:
        """Check if Gemini API is responding.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try embedding a simple test string
            await self.embed(["test"])
            return True
        except Exception:
            return False
