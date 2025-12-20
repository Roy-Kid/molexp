"""Utilities for ID generation and validation."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def generate_run_id() -> str:
    """Generate a unique run ID with timestamp and short UUID.

    Format: YYYYMMDD_HHMMSS_<short_uuid>
    Example: 20251129_181644_a3b2
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid4())[:4]
    return f"{timestamp}_{short_uuid}"


def generate_asset_id() -> str:
    """Generate a unique asset ID using UUID v4.

    Returns:
        UUID string, e.g., 'a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b'
    """
    return str(uuid4())


def validate_slug(slug: str, min_len: int = 3, max_len: int = 50) -> bool:
    """Validate a slug (project_id or experiment_id).

    Args:
        slug: String to validate
        min_len: Minimum length
        max_len: Maximum length

    Returns:
        True if valid, False otherwise
    """
    if not slug or not min_len <= len(slug) <= max_len:
        return False
    # Only lowercase letters, digits, and hyphens
    return bool(re.match(r"^[a-z0-9-]+$", slug))


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
