"""Ollama provider for local AI inference."""

import httpx
from typing import List, Optional

from app.core.providers.base import BaseProvider
from app.config import settings


class OllamaProvider(BaseProvider):
    """Ollama provider for local, free AI inference.

    Model configuration is loaded from settings (can be changed via environment variables):
    - Embedding model: OLLAMA_EMBEDDING_MODEL (default: nomic-embed-text)
    - LLM model: OLLAMA_LLM_MODEL (default: llama3.1:8b)
    - Embedding dimension: OLLAMA_EMBEDDING_DIMENSION (default: 768)
    """

    def __init__(self, base_url: Optional[str] = None):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL (default: from settings)
        """
        super().__init__(name="ollama")
        self.base_url = base_url or settings.ollama_url

        # Load model names from settings
        self.embed_model = settings.ollama_embedding_model
        self.llm_model = settings.ollama_llm_model
        self.dimension = settings.ollama_embedding_dimension
        self.timeout = 300

        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4)
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured embedding model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (dimension specified in settings)
        """
        embeddings = []

        for text in texts:
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
            except Exception as e:
                raise RuntimeError(f"Ollama embedding failed for text (len={len(text)}): {e}") from e

        return embeddings

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text using the configured LLM model.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            Generated text
        """
        from app.core.logging import get_logger
        logger = get_logger(__name__)

        # Retry logic for timeout errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False
                }

                if system:
                    payload["system"] = system

                logger.debug(
                    "ollama_generate_request",
                    model=self.llm_model,
                    base_url=self.base_url,
                    prompt_length=len(prompt),
                    has_system=system is not None,
                    attempt=attempt + 1,
                    max_retries=max_retries
                )

                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )

                logger.debug(
                    "ollama_generate_response",
                    status_code=response.status_code,
                    response_length=len(response.text),
                    headers=dict(response.headers)
                )

                response.raise_for_status()
                data = response.json()

                logger.debug(
                    "ollama_generate_parsed",
                    response_keys=list(data.keys()),
                    response_length=len(data.get("response", ""))
                )

                return data["response"]

            except (httpx.TimeoutException, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "ollama_generate_timeout_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e)
                    )
                    continue  # Retry
                else:
                    logger.error(
                        "ollama_generate_timeout_final",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    raise RuntimeError(f"Ollama LLM generation timed out after {max_retries} attempts: {str(e)}")

            except Exception as e:
                logger.error(
                    "ollama_generate_error",
                    model=self.llm_model,
                    base_url=self.base_url,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise RuntimeError(f"Ollama generation failed: {e}") from e

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension from settings (default: 768)
        """
        return self.dimension

    def get_cost_per_1k_tokens(self, operation: str = "embed") -> float:
        """Get cost per 1K tokens.

        Args:
            operation: "embed" or "generate"

        Returns:
            0.0 (Ollama is free - runs locally)
        """
        return 0.0

    async def health_check(self) -> bool:
        """Check if Ollama server is responding.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
