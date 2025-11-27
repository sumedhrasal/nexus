"""Local file system source connector."""

from pathlib import Path
from typing import AsyncGenerator, List, Optional
from app.core.sources.base import BaseSource
from app.core.entities import FileEntity
import logging

logger = logging.getLogger(__name__)


class LocalFileSource(BaseSource):
    """Local file system source connector."""

    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.html', '.json'}

    def __init__(self):
        """Initialize local file source."""
        super().__init__(name="local_files")

    async def fetch(
        self,
        path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> AsyncGenerator[FileEntity, None]:
        """Fetch files from local file system.

        Args:
            path: Directory or file path
            recursive: Recursively scan subdirectories
            extensions: File extensions to include (default: all supported)

        Yields:
            FileEntity instances
        """
        target_path = Path(path)
        allowed_extensions = set(extensions) if extensions else self.SUPPORTED_EXTENSIONS

        if not target_path.exists():
            logger.error(f"Path does not exist: {path}")
            return

        # Handle single file
        if target_path.is_file():
            if target_path.suffix in allowed_extensions:
                entity = await self._process_file(target_path)
                if entity:
                    yield entity
            return

        # Handle directory
        pattern = "**/*" if recursive else "*"
        for file_path in target_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in allowed_extensions:
                entity = await self._process_file(file_path)
                if entity:
                    yield entity

    async def _process_file(self, file_path: Path) -> Optional[FileEntity]:
        """Process a single file.

        Args:
            file_path: Path to file

        Returns:
            FileEntity or None if processing fails
        """
        try:
            # Read file content based on type
            content = await self._read_file(file_path)
            if not content:
                return None

            entity = FileEntity(
                entity_id=str(file_path.absolute()),
                entity_type="file",
                content=content,
                title=file_path.name,
                file_path=str(file_path.absolute()),
                file_type=file_path.suffix,
                file_size=file_path.stat().st_size,
                metadata={
                    "extension": file_path.suffix,
                    "directory": str(file_path.parent)
                }
            )
            return entity

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None

    async def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content.

        Args:
            file_path: Path to file

        Returns:
            File content as string
        """
        try:
            suffix = file_path.suffix.lower()

            if suffix in {'.txt', '.md', '.json', '.html'}:
                # Text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            elif suffix == '.pdf':
                # PDF files
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(str(file_path))
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    logger.warning("PyPDF2 not available, skipping PDF")
                    return None

            elif suffix == '.docx':
                # DOCX files
                try:
                    from docx import Document
                    doc = Document(str(file_path))
                    text = "\n".join([para.text for para in doc.paragraphs])
                    return text
                except ImportError:
                    logger.warning("python-docx not available, skipping DOCX")
                    return None

            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

    async def validate_config(self) -> bool:
        """Validate source configuration.

        Returns:
            Always True for local files
        """
        return True
