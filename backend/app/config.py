"""Application configuration."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://nexus:nexus@localhost:5432/nexus",
        description="PostgreSQL connection URL"
    )

    # Vector Database
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")

    # Cache
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")

    # AI Providers
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    gemini_api_key: Optional[str] = Field(default="dummy-key-replace-later", description="Google Gemini API key")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama server URL")

    # Ollama Model Configuration
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name (e.g., nomic-embed-text, mxbai-embed-large)"
    )
    ollama_llm_model: str = Field(
        default="qwen3-vl:4b",
        description="Ollama LLM model name (e.g., llama3.1:8b, llama3.2, mistral, qwen2.5)"
    )
    ollama_embedding_dimension: int = Field(
        default=768,
        description="Ollama embedding model dimension (768 for nomic-embed-text, 1024 for mxbai-embed-large)"
    )
    ollama_context_window: int = Field(
        default=4096,
        description="Ollama model context window size in tokens (llama3.1:8b=128k, llama3.2=128k, mistral=32k)"
    )

    # Cross-Encoder Re-ranking
    enable_reranking: bool = Field(
        default=True,
        description="Enable cross-encoder re-ranking for improved search relevance"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="HuggingFace model for re-ranking (bge-reranker-base, ms-marco-MiniLM-L-6-v2)"
    )
    reranker_top_k: int = Field(
        default=20,
        description="Number of top candidates to re-rank (higher = better quality, slower)"
    )

    # Source Integrations
    github_token: Optional[str] = Field(default=None, description="GitHub personal access token")
    gmail_client_id: Optional[str] = Field(default=None, description="Gmail OAuth client ID")
    gmail_client_secret: Optional[str] = Field(default=None, description="Gmail OAuth client secret")
    gmail_redirect_uri: str = Field(
        default="http://localhost:8000/auth/gmail/callback",
        description="Gmail OAuth redirect URI"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    json_logs: bool = Field(default=False, description="Enable JSON structured logging (for production)")

    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for JWT and encryption"
    )
    api_key_salt: str = Field(
        default="dev-salt-change-in-production",
        description="Salt for API key hashing"
    )

    # Performance
    max_workers: int = Field(default=4, description="Max worker threads")
    batch_size: int = Field(default=100, description="Batch size for embedding generation")
    chunk_size: int = Field(default=8192, description="Max tokens per chunk")
    chunk_overlap: int = Field(default=100, description="Token overlap between chunks")

    # Costs (for analytics)
    openai_embed_cost_per_1k: float = Field(default=0.0001, description="OpenAI embedding cost per 1K tokens")
    gemini_embed_cost_per_1k: float = Field(default=0.00001, description="Gemini embedding cost per 1K tokens")
    ollama_embed_cost_per_1k: float = Field(default=0.0, description="Ollama embedding cost (free)")

    # Provider Configuration
    embedding_provider_strategy: str = Field(
        default="cost",
        description="Provider selection strategy: local-first, cost, fallback"
    )
    llm_provider_strategy: str = Field(
        default="cost",
        description="LLM provider selection strategy"
    )


# Global settings instance
settings = Settings()
