"""Content hashing for change detection."""

import hashlib
import json
from typing import Any, Dict


def hash_content(content: str, metadata: Dict[str, Any] = None) -> str:
    """Generate SHA256 hash of content for change detection.

    Args:
        content: Text content
        metadata: Optional metadata to include in hash

    Returns:
        Hex digest of SHA256 hash
    """
    hasher = hashlib.sha256()

    # Hash content
    hasher.update(content.encode("utf-8"))

    # Hash metadata if provided (for structural changes)
    if metadata:
        # Sort keys for deterministic hashing
        metadata_str = json.dumps(metadata, sort_keys=True)
        hasher.update(metadata_str.encode("utf-8"))

    return hasher.hexdigest()


def hash_file(file_path: str) -> str:
    """Generate SHA256 hash of file contents.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256 hash
    """
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()
