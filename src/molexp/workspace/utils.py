"""Workspace utility functions for ID generation, slugification, and content hashing."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4


def generate_id() -> str:
    """Generate a unique 8-character hex run ID.

    Returns:
        8-character hex string, e.g., 'a3f2e8d9'
    """
    return uuid4().hex[:8]


def generate_asset_id() -> str:
    """Generate a unique asset ID using UUID.
    
    Returns:
        UUID string, e.g., 'a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b'
    """
    return str(uuid4())


def slugify(text: str, max_len: int = 50) -> str:
    """Convert text to a valid slug.
    
    Args:
        text: Input text
        max_len: Maximum length
        
    Returns:
        Slugified string
    """
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove non-alphanumeric characters except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    # Truncate to max length
    return slug[:max_len]


def compute_content_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of file content.
    
    Args:
        path: Path to file
        algorithm: Hash algorithm (default: sha256)
        
    Returns:
        Hash string with algorithm prefix, e.g., 'sha256:a3b4c5d6...'
    """
    hasher = hashlib.new(algorithm)
    
    with open(path, "rb") as f:
        # Read in chunks for large files
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    return f"{algorithm}:{hasher.hexdigest()}"


