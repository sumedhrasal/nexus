"""Ollama provider for local AI inference."""

import httpx
from typing import List, Optional

from app.core.providers.base import BaseProvider
from app.config import settings


class OllamaProvider(BaseProvider):
    """Ollama provider for local, free AI inference.

    Uses:
    - nomic-embed-text for embeddings (768 dimensions)
    - llama3.1 for text generation
    """

    def __init__(self, base_url: Optional[str] = None):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL (default: from settings)
        """
        super().__init__(name="ollama")
        self.base_url = base_url or settings.ollama_url
        # Replace host.docker.internal with IPv4 address to avoid IPv6 connection issues
        if "host.docker.internal" in self.base_url:
            self.base_url = self.base_url.replace("host.docker.internal", "192.168.65.254")
        self.embed_model = "nomic-embed-text"  # 768d, fast, good quality
        self.llm_model = "llama3.1:8b"  # Llama 3.1 8B model
        self.client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.dimension = 768

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using nomic-embed-text.

        Args:
            texts: List of texts to embed

        Returns:
            List of 768-dimensional embedding vectors
        """
        embeddings = []

        for text in texts:
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
            except Exception as e:
                raise RuntimeError(f"Ollama embedding failed for text (len={len(text)}): {e}") from e

        return embeddings

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text using llama3.1.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }

            if system:
                payload["system"] = system

            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            768 (nomic-embed-text dimension)
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
