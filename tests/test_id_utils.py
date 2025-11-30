"""Tests for ID utilities."""

import pytest
from pathlib import Path
import tempfile
from molexp.id_utils import (
    generate_run_id,
    generate_asset_id,
    validate_slug,
    compute_content_hash,
    slugify,
)


def test_generate_run_id():
    """Test run ID generation."""
    run_id = generate_run_id()
    
    # Should have format: YYYYMMDD_HHMMSS_xxxx
    parts = run_id.split("_")
    assert len(parts) == 3
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 6  # HHMMSS
    assert len(parts[2]) == 4  # short UUID
    
    # Should be unique
    run_id2 = generate_run_id()
    assert run_id != run_id2


def test_generate_asset_id():
    """Test asset ID generation."""
    asset_id = generate_asset_id()
    
    # Should be a valid UUID
    assert len(asset_id) == 36
    assert asset_id.count("-") == 4
    
    # Should be unique
    asset_id2 = generate_asset_id()
    assert asset_id != asset_id2


def test_validate_slug():
    """Test slug validation."""
    # Valid slugs
    assert validate_slug("valid-slug")
    assert validate_slug("project123")
    assert validate_slug("my-project-name")
    
    # Invalid slugs
    assert not validate_slug("AB")  # Too short
    assert not validate_slug("Invalid_Slug")  # Underscore
    assert not validate_slug("Invalid Slug")  # Space
    assert not validate_slug("Invalid-Slug")  # Uppercase
    assert not validate_slug("")  # Empty
    assert not validate_slug("a" * 51)  # Too long


def test_compute_content_hash():
    """Test content hash computation."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = Path(f.name)
    
    try:
        hash1 = compute_content_hash(temp_path)
        assert hash1.startswith("sha256:")
        assert len(hash1) > 10
        
        # Same content should produce same hash
        hash2 = compute_content_hash(temp_path)
        assert hash1 == hash2
    finally:
        temp_path.unlink()


def test_slugify():
    """Test slugify function."""
    assert slugify("My Project Name") == "my-project-name"
    assert slugify("Project_123") == "project-123"
    assert slugify("  Spaces  ") == "spaces"
    assert slugify("Special!@#Characters") == "specialcharacters"
    assert slugify("Multiple---Hyphens") == "multiple-hyphens"
    assert slugify("A" * 60, max_len=50) == "a" * 50
