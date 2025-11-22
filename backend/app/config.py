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
    ollama_embedding_model: Optional[str] = Field(
        default=None,
        description="Ollama embedding model name (e.g., nomic-embed-text, mxbai-embed-large)"
    )
    ollama_llm_model: Optional[str] = Field(
        default=None,
        description="Ollama LLM model name (e.g., llama3.1:8b, llama3.2, mistral, qwen2.5)"
    )
    ollama_embedding_dimension: Optional[int] = Field(
        default=None,
        description="Ollama embedding model dimension (768 for nomic-embed-text, 1024 for mxbai-embed-large)"
    )
    ollama_context_window: Optional[int] = Field(
        default=None,
        description="Ollama model context window size in tokens (llama3.1:8b=128k, llama3.2=128k, mistral=32k)"
    )

    # Gemini Model Configuration
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model name"
    )
    gemini_llm_model: str = Field(
        default="gemini-2.5-flash-exp",
        description="Gemini LLM model name (gemini-2.5-flash-exp, gemini-1.5-pro)"
    )
    gemini_embedding_dimension: int = Field(
        default=768,
        description="Gemini embedding model dimension (text-embedding-004=768)"
    )
    gemini_context_window: int = Field(
        default=32768,
        description="Gemini model context window size in tokens (gemini-2.5-flash=32k, gemini-1.5-pro=2M)"
    )

    # OpenAI Model Configuration
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name (text-embedding-3-small, text-embedding-3-large)"
    )
    openai_llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model name (gpt-4o-mini, gpt-4o)"
    )
    openai_embedding_dimension: int = Field(
        default=1536,
        description="OpenAI embedding model dimension (small=1536, large=3072)"
    )
    openai_context_window: int = Field(
        default=8192,
        description="OpenAI model context window size in tokens (gpt-4o-mini=128k, gpt-4o=128k)"
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
    reranker_max_length: int = Field(
        default=512,
        description="Maximum token length for cross-encoder model (bge-reranker-base=512, ms-marco=512)"
    )
    reranker_top_k: int = Field(
        default=20,
        description="Number of top candidates to re-rank (higher = better quality, slower)"
    )

    # Qdrant Storage
    qdrant_max_payload_size: int = Field(
        default=5000,
        description="Maximum character length for parent_content in Qdrant payloads (prevents HTTP timeouts)"
    )

    # Metadata Extraction
    metadata_prompt_overhead: int = Field(
        default=800,
        description="Reserved tokens for system/user prompts and response in metadata extraction (system~200, user~100, response~500)"
    )
    metadata_chunk_max_tokens: int = Field(
        default=500,
        description="Maximum tokens for quick chunk metadata extraction (smaller = faster)"
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
