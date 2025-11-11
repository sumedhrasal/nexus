"""Provider router with fallback strategy."""

from typing import List, Optional, Tuple
import logging

from app.core.providers.base import BaseProvider
from app.core.providers.ollama import OllamaProvider
from app.core.providers.gemini import GeminiProvider
from app.core.providers.openai_provider import OpenAIProvider
from app.core.embedding_cache import get_embedding_cache, EmbeddingCache
from app.config import settings

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Router for managing multiple AI providers with fallback.

    Strategies:
    - local-first: Try Ollama first (free), then cloud providers
    - cost: Use cheapest provider first
    - fallback: Try providers in order until success
    """

    def __init__(self, strategy: str = "local-first", use_cache: bool = True):
        """Initialize provider router.

        Args:
            strategy: Selection strategy (local-first, cost, fallback)
            use_cache: Whether to use embedding cache (default: True)
        """
        self.strategy = strategy
        self.use_cache = use_cache
        self.providers: List[BaseProvider] = []
        self.cache: Optional[EmbeddingCache] = None

        if self.use_cache:
            self.cache = get_embedding_cache()
            logger.info("✓ Embedding cache enabled")

        self._initialize_providers()
        self._sort_providers()

    def _initialize_providers(self):
        """Initialize available providers based on configuration."""
        # Try Ollama (always available, no API key needed)
        try:
            ollama = OllamaProvider()
            self.providers.append(ollama)
            logger.info("✓ Ollama provider initialized")
        except Exception as e:
            logger.warning(f"✗ Ollama provider failed: {e}")

        # Try Gemini
        try:
            if settings.gemini_api_key and settings.gemini_api_key != "dummy-key-replace-later":
                gemini = GeminiProvider()
                self.providers.append(gemini)
                logger.info("✓ Gemini provider initialized")
        except Exception as e:
            logger.warning(f"✗ Gemini provider failed: {e}")

        # Try OpenAI
        try:
            if settings.openai_api_key:
                openai = OpenAIProvider()
                self.providers.append(openai)
                logger.info("✓ OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"✗ OpenAI provider failed: {e}")

        if not self.providers:
            raise RuntimeError(
                "No AI providers available! Please configure at least one: "
                "OPENAI_API_KEY, GEMINI_API_KEY, or ensure Ollama is running."
            )

        logger.info(f"Initialized {len(self.providers)} provider(s): {[p.name for p in self.providers]}")

    def _sort_providers(self):
        """Sort providers based on strategy."""
        if self.strategy == "cost":
            # Sort by cost (ascending)
            self.providers.sort(key=lambda p: p.get_cost_per_1k_tokens("embed"))
        elif self.strategy == "local-first":
            # Ollama first, then by cost
            local = [p for p in self.providers if isinstance(p, OllamaProvider)]
            cloud = [p for p in self.providers if not isinstance(p, OllamaProvider)]
            cloud.sort(key=lambda p: p.get_cost_per_1k_tokens("embed"))
            self.providers = local + cloud
        # else: fallback strategy - use current order

        logger.info(f"Provider order ({self.strategy}): {[p.name for p in self.providers]}")

    async def embed(
        self,
        texts: List[str],
        required_dimension: Optional[int] = None
    ) -> Tuple[List[List[float]], str]:
        """Generate embeddings with provider fallback and caching.

        Args:
            texts: List of texts to embed
            required_dimension: Required embedding dimension (filters providers)

        Returns:
            Tuple of (embeddings, provider_name)

        Raises:
            RuntimeError: If all providers fail
        """
        # Check cache if enabled
        if self.use_cache and self.cache:
            # Get first provider name for cache key
            provider_name = self.providers[0].name if self.providers else "default"

            # Try to get from cache
            cached_embeddings, missing_indices = await self.cache.get_batch(texts, provider_name)

            # If all cached, return immediately
            if not missing_indices:
                logger.info(f"✓ All {len(texts)} embeddings from cache")
                return cached_embeddings, f"{provider_name} (cached)"

            # Some or all not cached, need to generate
            logger.debug(f"Cache miss for {len(missing_indices)}/{len(texts)} embeddings")
        else:
            cached_embeddings = [None] * len(texts)
            missing_indices = list(range(len(texts)))

        # Filter providers by dimension if specified
        providers = self.providers
        if required_dimension:
            providers = [p for p in providers if p.get_embedding_dimension() == required_dimension]
            if not providers:
                raise ValueError(
                    f"No providers available for dimension {required_dimension}. "
                    f"Available dimensions: {[p.get_embedding_dimension() for p in self.providers]}"
                )

        # Get texts that need embedding
        texts_to_embed = [texts[i] for i in missing_indices]

        # Try providers with fallback
        errors = []
        for provider in providers:
            try:
                logger.debug(f"Trying {provider.name} for {len(texts_to_embed)} embeddings...")
                new_embeddings = await provider.embed(texts_to_embed)
                logger.info(f"✓ {provider.name} succeeded for {len(texts_to_embed)} texts")

                # Fill in the new embeddings
                final_embeddings = cached_embeddings.copy()
                for idx, embedding in zip(missing_indices, new_embeddings):
                    final_embeddings[idx] = embedding

                # Cache the newly generated embeddings
                if self.use_cache and self.cache:
                    await self.cache.set_batch(texts_to_embed, new_embeddings, provider.name)
                    logger.debug(f"Cached {len(new_embeddings)} new embeddings")

                return final_embeddings, provider.name

            except Exception as e:
                error_msg = f"{provider.name} failed: {str(e)[:100]}"
                logger.warning(f"✗ {error_msg}")
                errors.append(error_msg)
                continue

        # All providers failed
        raise RuntimeError(
            f"All embedding providers failed for {len(texts_to_embed)} texts. Errors:\n" +
            "\n".join(f"- {e}" for e in errors)
        )

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate text with provider fallback.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            Tuple of (generated_text, provider_name)

        Raises:
            RuntimeError: If all providers fail
        """
        errors = []
        for provider in self.providers:
            try:
                logger.debug(f"Trying {provider.name} for generation...")
                text = await provider.generate(prompt, system)
                logger.info(f"✓ {provider.name} succeeded for generation")
                return text, provider.name
            except Exception as e:
                error_msg = f"{provider.name} failed: {str(e)[:100]}"
                logger.warning(f"✗ {error_msg}")
                errors.append(error_msg)
                continue

        # All providers failed
        raise RuntimeError(
            f"All LLM providers failed. Errors:\n" +
            "\n".join(f"- {e}" for e in errors)
        )

    async def health_check_all(self) -> dict:
        """Check health of all providers.

        Returns:
            Dict mapping provider names to health status
        """
        results = {}
        for provider in self.providers:
            try:
                healthy = await provider.health_check()
                results[provider.name] = "healthy" if healthy else "unhealthy"
            except Exception as e:
                results[provider.name] = f"error: {str(e)[:50]}"

        return results

    def get_primary_dimension(self) -> int:
        """Get dimension of primary (first) provider.

        Returns:
            Embedding dimension
        """
        if not self.providers:
            raise RuntimeError("No providers available")
        return self.providers[0].get_embedding_dimension()

    def get_provider_info(self) -> List[dict]:
        """Get information about all providers.

        Returns:
            List of provider info dicts
        """
        return [
            {
                "name": p.name,
                "dimension": p.get_embedding_dimension(),
                "cost_per_1k": p.get_cost_per_1k_tokens("embed"),
                "type": p.__class__.__name__
            }
            for p in self.providers
        ]

    def get_cache_stats(self) -> Optional[dict]:
        """Get embedding cache statistics.

        Returns:
            Cache stats dict if cache is enabled, None otherwise
        """
        if self.use_cache and self.cache:
            return self.cache.get_stats()
        return None


# Global router instance
_router: Optional[ProviderRouter] = None


def get_provider_router(strategy: Optional[str] = None) -> ProviderRouter:
    """Get or create global provider router.

    Args:
        strategy: Provider selection strategy (default: from settings)

    Returns:
        ProviderRouter instance
    """
    global _router
    if _router is None:
        strategy = strategy or settings.embedding_provider_strategy
        _router = ProviderRouter(strategy=strategy)
    return _router
