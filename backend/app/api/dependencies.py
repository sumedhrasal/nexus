"""API dependencies for dependency injection."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.storage.postgres import get_db as get_db_session
from app.storage.qdrant import get_qdrant, QdrantStorage
from app.core.providers.router import get_provider_router, ProviderRouter


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async for session in get_db_session():
        yield session


async def get_qdrant_client() -> QdrantStorage:
    """Get Qdrant client."""
    return get_qdrant()


async def get_embedding_router() -> ProviderRouter:
    """Get provider router."""
    return get_provider_router()
