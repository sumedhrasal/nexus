"""Tests for directory scanner service."""

import tempfile
import pytest
from pathlib import Path

from app.sources.directory_scanner import DirectoryScanner


class TestDirectoryScanner:
    """Test DirectoryScanner class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create directory structure
            (base / "docs").mkdir()
            (base / "src").mkdir()
            (base / "tests").mkdir()
            (base / "node_modules").mkdir()
            (base / ".git").mkdir()

            # Create text files
            (base / "README.md").write_text("# README")
            (base / "docs" / "guide.md").write_text("# Guide")
            (base / "docs" / "api.md").write_text("# API")
            (base / "src" / "main.py").write_text("print('hello')")
            (base / "src" / "utils.py").write_text("def helper(): pass")
            (base / "tests" / "test_main.py").write_text("def test(): pass")

            # Create files to exclude
            (base / "node_modules" / "package.json").write_text("{}")
            (base / ".git" / "config").write_text("[core]")

            # Create binary file (simulate with null bytes)
            (base / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

            # Create large file
            large_content = "x" * (15 * 1024 * 1024)  # 15 MB
            (base / "large.txt").write_text(large_content)

            # Create empty file
            (base / "empty.txt").write_text("")

            yield base

    def test_scanner_initialization(self, temp_dir):
        """Test scanner can be initialized."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=[".md"],
            exclude_paths=[".git", "node_modules"]
        )
        assert scanner.base_path == temp_dir.resolve()
        assert scanner.file_types == [".md"]

    def test_scanner_invalid_directory(self):
        """Test scanner raises error for invalid directory."""
        with pytest.raises(ValueError, match="does not exist"):
            DirectoryScanner(base_path="/nonexistent/path")

    def test_scanner_file_instead_of_directory(self, temp_dir):
        """Test scanner raises error when path is a file."""
        file_path = temp_dir / "README.md"
        with pytest.raises(ValueError, match="not a directory"):
            DirectoryScanner(base_path=str(file_path))

    def test_scan_all_text_files(self, temp_dir):
        """Test scanning all text files."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=None,  # All text files
            exclude_paths=[".git", "node_modules"]
        )
        files = scanner.scan(max_files=100)

        # Should find: README.md, guide.md, api.md, main.py, utils.py, test_main.py, empty.txt
        # Should NOT find: image.png (binary), large.txt (>10MB default), node_modules/*, .git/*
        file_names = [f["relative_path"].name for f in files]

        assert "README.md" in file_names
        assert "guide.md" in file_names
        assert "api.md" in file_names
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" in file_names
        assert "empty.txt" in file_names

        assert "image.png" not in file_names
        assert "large.txt" not in file_names
        assert "package.json" not in file_names
        assert "config" not in file_names

    def test_scan_filter_by_file_type(self, temp_dir):
        """Test filtering by file extension."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=[".md"],
            exclude_paths=[".git", "node_modules"]
        )
        files = scanner.scan(max_files=100)

        file_names = [f["relative_path"].name for f in files]

        assert "README.md" in file_names
        assert "guide.md" in file_names
        assert "api.md" in file_names

        # Should not include Python files
        assert "main.py" not in file_names
        assert "utils.py" not in file_names

    def test_scan_filter_by_include_paths(self, temp_dir):
        """Test filtering by include paths."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=None,
            include_paths=["docs/"],
            exclude_paths=[".git", "node_modules"]
        )
        files = scanner.scan(max_files=100)

        file_names = [f["relative_path"].name for f in files]

        # Should only include files in docs/
        assert "guide.md" in file_names
        assert "api.md" in file_names

        # Should not include files from other directories
        assert "README.md" not in file_names
        assert "main.py" not in file_names

    def test_scan_exclude_paths(self, temp_dir):
        """Test excluding paths."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=None,
            exclude_paths=[".git", "node_modules", "tests"]
        )
        files = scanner.scan(max_files=100)

        file_names = [f["relative_path"].name for f in files]

        # Should not include files from excluded directories
        assert "test_main.py" not in file_names
        assert "config" not in file_names
        assert "package.json" not in file_names

    def test_scan_max_file_size(self, temp_dir):
        """Test max file size filtering."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=None,
            exclude_paths=[".git", "node_modules"],
            max_file_size_mb=1  # 1 MB limit
        )
        files = scanner.scan(max_files=100)

        file_names = [f["relative_path"].name for f in files]

        # Large file should be excluded
        assert "large.txt" not in file_names

        # Normal files should be included
        assert "README.md" in file_names

    def test_scan_max_files_limit(self, temp_dir):
        """Test max files limit."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=None,
            exclude_paths=[".git", "node_modules"]
        )
        files = scanner.scan(max_files=3)

        # Should stop after 3 files
        assert len(files) == 3

    def test_scan_binary_file_detection(self, temp_dir):
        """Test that binary files are excluded."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=None,
            exclude_paths=[".git", "node_modules"]
        )
        files = scanner.scan(max_files=100)

        file_names = [f["relative_path"].name for f in files]

        # Binary PNG file should be excluded
        assert "image.png" not in file_names

    def test_scan_returns_correct_metadata(self, temp_dir):
        """Test that scan returns correct file metadata."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=[".md"],
            exclude_paths=[".git", "node_modules"]
        )
        files = scanner.scan(max_files=100)

        assert len(files) > 0

        # Check first file has expected keys
        first_file = files[0]
        assert "path" in first_file
        assert "relative_path" in first_file
        assert "size_mb" in first_file

        # Check path is absolute Path object
        assert isinstance(first_file["path"], Path)
        assert first_file["path"].is_absolute()

        # Check relative path is relative to base
        assert isinstance(first_file["relative_path"], Path)
        assert not first_file["relative_path"].is_absolute()

        # Check size is a number
        assert isinstance(first_file["size_mb"], (int, float))
        assert first_file["size_mb"] >= 0

    def test_extract_git_metadata_no_git(self, temp_dir):
        """Test git metadata extraction when not a git repo."""
        # Remove .git directory
        import shutil
        shutil.rmtree(temp_dir / ".git")

        scanner = DirectoryScanner(base_path=str(temp_dir))
        metadata = scanner.extract_git_metadata()

        assert metadata is None

    def test_extract_git_metadata_with_git(self, temp_dir):
        """Test git metadata extraction from a git repo."""
        # Initialize a real git repo
        import subprocess

        try:
            subprocess.run(
                ["git", "init"],
                cwd=temp_dir,
                capture_output=True,
                timeout=5,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=temp_dir,
                capture_output=True,
                timeout=5,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=temp_dir,
                capture_output=True,
                timeout=5,
                check=True
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=temp_dir,
                capture_output=True,
                timeout=5,
                check=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=temp_dir,
                capture_output=True,
                timeout=5,
                check=True
            )

            scanner = DirectoryScanner(base_path=str(temp_dir))
            metadata = scanner.extract_git_metadata()

            # Should extract metadata
            assert metadata is not None
            assert "repo_name" in metadata
            assert "commit_sha" in metadata

            # Repo name should be directory name (no remote configured)
            assert metadata["repo_name"] == temp_dir.name

            # Commit SHA should be 8 characters
            assert len(metadata["commit_sha"]) == 8

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Git not available or command failed")

    def test_is_text_file_common_extensions(self, temp_dir):
        """Test text file detection for common extensions."""
        scanner = DirectoryScanner(base_path=str(temp_dir))

        # Create files with various extensions
        test_files = {
            "test.py": True,
            "test.js": True,
            "test.md": True,
            "test.json": True,
            "test.yaml": True,
            "test.txt": True,
            "test.html": True,
            "test.css": True,
            "Dockerfile": True,
            "Makefile": True,
            "README": True,
        }

        for filename, expected_text in test_files.items():
            filepath = temp_dir / filename
            if expected_text:
                filepath.write_text("content")
            else:
                filepath.write_bytes(b"\x00\x01\x02")

            result = scanner._is_text_file(filepath)
            assert result == expected_text, f"{filename} should be {'text' if expected_text else 'binary'}"

    def test_multiple_file_types(self, temp_dir):
        """Test filtering by multiple file types."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=[".md", ".py"],
            exclude_paths=[".git", "node_modules"]
        )
        files = scanner.scan(max_files=100)

        file_names = [f["relative_path"].name for f in files]

        # Should include both .md and .py files
        assert "README.md" in file_names
        assert "main.py" in file_names

        # Should not include .txt files
        assert "empty.txt" not in file_names

    def test_combined_filters(self, temp_dir):
        """Test combining multiple filters."""
        scanner = DirectoryScanner(
            base_path=str(temp_dir),
            file_types=[".py"],
            include_paths=["src/"],
            exclude_paths=[".git", "node_modules"],
            max_file_size_mb=5
        )
        files = scanner.scan(max_files=100)

        file_names = [f["relative_path"].name for f in files]

        # Should only include .py files from src/
        assert "main.py" in file_names
        assert "utils.py" in file_names

        # Should not include .py files from other directories
        assert "test_main.py" not in file_names

        # Should not include non-.py files from src/
        assert "README.md" not in file_names
