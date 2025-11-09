"""Integration test for source connectors."""

import pytest
import tempfile
from pathlib import Path
from app.core.sources.local_files import LocalFileSource


@pytest.mark.asyncio
async def test_local_file_source():
    """Test local file source connector."""

    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files
        (tmp_path / "test1.txt").write_text("This is test file 1 content.")
        (tmp_path / "test2.md").write_text("# Markdown Test\n\nThis is test file 2.")
        (tmp_path / "test3.json").write_text('{"key": "value"}')
        (tmp_path / "ignored.xyz").write_text("This should be ignored")

        # Test source
        source = LocalFileSource()
        entities = []

        async for entity in source.fetch(path=str(tmp_path), recursive=False):
            entities.append(entity)

        # Verify results
        assert len(entities) == 3  # Only .txt, .md, .json
        assert all(entity.entity_type == "file" for entity in entities)
        assert all(entity.content for entity in entities)
        assert all(entity.file_path for entity in entities)

        # Verify specific content
        txt_entity = next(e for e in entities if e.title == "test1.txt")
        assert "test file 1 content" in txt_entity.content

        md_entity = next(e for e in entities if e.title == "test2.md")
        assert "Markdown Test" in md_entity.content
