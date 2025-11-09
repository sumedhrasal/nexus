"""Base source connector interface."""

from abc import ABC, abstractmethod
from typing import List, AsyncGenerator
from app.core.entities import BaseEntity


class BaseSource(ABC):
    """Abstract base class for source connectors."""

    def __init__(self, name: str):
        """Initialize source connector.

        Args:
            name: Source connector name
        """
        self.name = name

    @abstractmethod
    async def fetch(self, **kwargs) -> AsyncGenerator[BaseEntity, None]:
        """Fetch entities from source.

        Args:
            **kwargs: Source-specific parameters

        Yields:
            BaseEntity instances
        """
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate source configuration.

        Returns:
            True if configuration is valid
        """
        pass
