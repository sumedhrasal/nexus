"""Directory scanning service for local file ingestion."""

import os
import subprocess
import mimetypes
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class DirectoryScanner:
    """Scan local directories with filtering capabilities."""

    def __init__(
        self,
        base_path: str,
        file_types: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_file_size_mb: int = 10
    ):
        """Initialize directory scanner.

        Args:
            base_path: Absolute path to the directory to scan
            file_types: File extensions to include (e.g., ['.md', '.py']). None = all text files
            include_paths: Path patterns to include (e.g., ['docs/', 'src/']). None = all paths
            exclude_paths: Path patterns to exclude (e.g., ['.git', 'node_modules'])
            max_file_size_mb: Maximum file size in MB to process
        """
        self.base_path = Path(base_path).resolve()
        self.file_types = file_types or []
        self.include_paths = include_paths or []
        self.exclude_paths = exclude_paths or [".git", "node_modules", "__pycache__", ".venv", "venv"]
        self.max_file_size_mb = max_file_size_mb

        if not self.base_path.exists():
            raise ValueError(f"Directory does not exist: {base_path}")

        if not self.base_path.is_dir():
            raise ValueError(f"Path is not a directory: {base_path}")

    def scan(self, max_files: int = 1000) -> List[Dict[str, Any]]:
        """Scan directory and return filtered files.

        Args:
            max_files: Maximum number of files to return

        Returns:
            List of file info dicts with keys: path, relative_path, size_mb
        """
        files = []

        logger.info(
            "directory_scan_started",
            base_path=str(self.base_path),
            file_types=self.file_types,
            include_paths=self.include_paths,
            exclude_paths=self.exclude_paths
        )

        for file_path in self.base_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Apply filters
            if not self._should_include(file_path):
                continue

            # Check file size
            try:
                size_bytes = file_path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)

                if size_mb > self.max_file_size_mb:
                    logger.debug(
                        "file_too_large",
                        path=str(file_path.relative_to(self.base_path)),
                        size_mb=size_mb,
                        max_size_mb=self.max_file_size_mb
                    )
                    continue
            except OSError as e:
                logger.warning(f"Cannot stat file {file_path}: {e}")
                continue

            # Check if text file (skip binaries)
            if not self._is_text_file(file_path):
                logger.debug(
                    "binary_file_skipped",
                    path=str(file_path.relative_to(self.base_path))
                )
                continue

            files.append({
                "path": file_path,
                "relative_path": file_path.relative_to(self.base_path),
                "size_mb": round(size_mb, 3)
            })

            # Stop if we've reached max_files
            if len(files) >= max_files:
                logger.warning(
                    "max_files_reached",
                    max_files=max_files,
                    message="Stopped scanning after reaching max_files limit"
                )
                break

        logger.info(
            "directory_scan_completed",
            files_found=len(files),
            base_path=str(self.base_path)
        )

        return files

    def _should_include(self, file_path: Path) -> bool:
        """Check if file matches include/exclude filters."""
        rel_path = str(file_path.relative_to(self.base_path))

        # Exclude patterns (highest priority)
        for pattern in self.exclude_paths:
            if pattern in rel_path:
                return False

        # File type filter
        if self.file_types and file_path.suffix not in self.file_types:
            return False

        # Include patterns (if specified)
        if self.include_paths:
            return any(pattern in rel_path for pattern in self.include_paths)

        return True

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is text (not binary).

        Uses multiple heuristics:
        1. MIME type detection
        2. Common text file extensions
        3. Binary detection (null bytes in first 1KB)
        """
        # Check by MIME type first
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith("text"):
            return True

        # Common text extensions not always in mimetypes
        text_extensions = {
            ".md", ".txt", ".py", ".js", ".jsx", ".ts", ".tsx",
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".sh", ".bash", ".env", ".gitignore", ".dockerfile",
            ".html", ".htm", ".css", ".scss", ".sass", ".less",
            ".xml", ".csv", ".sql", ".graphql", ".proto",
            ".go", ".rs", ".rb", ".php", ".java", ".c", ".cpp",
            ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
            ".r", ".R", ".m", ".pl", ".vim", ".lua"
        }
        if file_path.suffix.lower() in text_extensions:
            return True

        # Files without extensions that are typically text
        if not file_path.suffix and file_path.name in {
            "Dockerfile", "Makefile", "LICENSE", "README",
            "CHANGELOG", "CONTRIBUTING", "AUTHORS"
        }:
            return True

        # Read first 1024 bytes to check for null bytes (binary indicator)
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                # Empty files are considered text
                if not chunk:
                    return True
                # Presence of null bytes indicates binary
                return b"\x00" not in chunk
        except Exception as e:
            logger.warning(f"Cannot read file {file_path} for binary check: {e}")
            return False

    def extract_git_metadata(self) -> Optional[Dict[str, str]]:
        """Extract git repository name and current commit SHA.

        Returns:
            Dict with 'repo_name' and 'commit_sha' keys, or None if not a git repo
        """
        git_dir = self.base_path / ".git"
        if not git_dir.exists():
            logger.debug("no_git_metadata", reason="No .git directory found")
            return None

        try:
            # Get repo name from remote URL
            result = subprocess.run(
                ["git", "-C", str(self.base_path), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                repo_url = result.stdout.strip()
                # Extract repo name from URL (handles both HTTPS and SSH)
                # https://github.com/owner/repo.git -> repo
                # git@github.com:owner/repo.git -> repo
                repo_name = repo_url.split("/")[-1].replace(".git", "")
            else:
                # Fallback to directory name if no remote
                repo_name = self.base_path.name

            # Get current commit SHA
            result = subprocess.run(
                ["git", "-C", str(self.base_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                commit_sha = result.stdout.strip()[:8]  # Short SHA (8 chars)
            else:
                logger.warning("git_sha_extraction_failed")
                return None

            logger.info(
                "git_metadata_extracted",
                repo_name=repo_name,
                commit_sha=commit_sha
            )

            return {
                "repo_name": repo_name,
                "commit_sha": commit_sha
            }
        except subprocess.TimeoutExpired:
            logger.warning("git_command_timeout")
            return None
        except Exception as e:
            logger.warning(f"git_metadata_extraction_error: {e}")
            return None
