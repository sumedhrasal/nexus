"""GitHub source connector."""

from typing import AsyncGenerator, Optional, List
from github import Github, GithubException
from app.core.sources.base import BaseSource
from app.core.entities import FileEntity
from app.config import settings
import logging
import base64

logger = logging.getLogger(__name__)


class GitHubSource(BaseSource):
    """GitHub repository source connector."""

    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c'}

    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub source.

        Args:
            token: GitHub personal access token
        """
        super().__init__(name="github")
        self.token = token or settings.github_token
        self.client: Optional[Github] = None

    async def fetch(
        self,
        repo_name: str,
        branch: str = "main",
        path: str = "",
        extensions: Optional[List[str]] = None
    ) -> AsyncGenerator[FileEntity, None]:
        """Fetch files from GitHub repository.

        Args:
            repo_name: Repository name (owner/repo)
            branch: Branch name
            path: Path within repository
            extensions: File extensions to include

        Yields:
            FileEntity instances
        """
        if not await self.validate_config():
            logger.error("GitHub token not configured")
            return

        try:
            self.client = Github(self.token)
            repo = self.client.get_repo(repo_name)
            allowed_extensions = set(extensions) if extensions else self.SUPPORTED_EXTENSIONS

            # Get repository contents
            contents = repo.get_contents(path, ref=branch)
            if not isinstance(contents, list):
                contents = [contents]

            # Process files recursively
            async for entity in self._process_contents(repo, contents, allowed_extensions, branch):
                yield entity

        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch from GitHub: {e}")
        finally:
            if self.client:
                self.client.close()

    async def _process_contents(
        self,
        repo,
        contents,
        allowed_extensions: set,
        branch: str
    ) -> AsyncGenerator[FileEntity, None]:
        """Process repository contents recursively.

        Args:
            repo: GitHub repository object
            contents: List of content objects
            allowed_extensions: Allowed file extensions
            branch: Branch name

        Yields:
            FileEntity instances
        """
        for content in contents:
            if content.type == "dir":
                # Recursively process directory
                subcontents = repo.get_contents(content.path, ref=branch)
                async for entity in self._process_contents(repo, subcontents, allowed_extensions, branch):
                    yield entity

            elif content.type == "file":
                # Process file
                if any(content.name.endswith(ext) for ext in allowed_extensions):
                    entity = await self._process_file(content, repo.full_name)
                    if entity:
                        yield entity

    async def _process_file(self, content, repo_name: str) -> Optional[FileEntity]:
        """Process a single file.

        Args:
            content: GitHub content object
            repo_name: Repository name

        Returns:
            FileEntity or None
        """
        try:
            # Decode content
            if content.encoding == "base64":
                file_content = base64.b64decode(content.content).decode('utf-8')
            else:
                file_content = content.decoded_content.decode('utf-8')

            entity = FileEntity(
                entity_id=content.sha,
                entity_type="file",
                content=file_content,
                title=content.name,
                file_path=content.path,
                file_type=content.name.split('.')[-1] if '.' in content.name else None,
                file_size=content.size,
                metadata={
                    "repo": repo_name,
                    "url": content.html_url,
                    "sha": content.sha
                }
            )
            return entity

        except Exception as e:
            logger.error(f"Failed to process GitHub file {content.path}: {e}")
            return None

    async def validate_config(self) -> bool:
        """Validate GitHub token.

        Returns:
            True if token is configured
        """
        return bool(self.token)
