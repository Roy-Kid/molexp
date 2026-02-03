"""Tests for resource abstraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from molexp.workspace.resource import FileResource, FolderResource, AssetResource


def test_file_resource_basic():
    """Test basic FileResource operations."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test content")
        temp_path = Path(f.name)
    
    try:
        # Create resource
        resource = FileResource(f"file://{temp_path}")
        
        # Test properties
        assert resource.uri == f"file://{temp_path}"
        assert resource.path == temp_path
        assert resource.exists()
        
        # Test read
        content = resource.read()
        assert content == b"test content"
        
        # Test write
        resource.write(b"new content")
        assert resource.read() == b"new content"
    finally:
        temp_path.unlink()


def test_folder_resource_basic():
    """Test basic FolderResource operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create some files
        (tmppath / "file1.txt").write_text("content1")
        (tmppath / "file2.txt").write_text("content2")
        (tmppath / "subdir").mkdir()
        (tmppath / "subdir" / "file3.txt").write_text("content3")
        
        # Create resource
        resource = FolderResource(f"file://{tmppath}")
        
        # Test properties
        assert resource.uri == f"file://{tmppath}"
        assert resource.path == tmppath
        assert resource.exists()
        
        # Test list_files
        files = resource.list_files()
        assert len(files) == 3
        assert all(f.is_file() for f in files)
        
        # Test read/write raise errors
        with pytest.raises(NotImplementedError):
            resource.read()
        with pytest.raises(NotImplementedError):
            resource.write(b"data")


def test_asset_resource_basic():
    """Test basic AssetResource operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir)
        asset_id = "test123"
        
        # Create asset directory structure
        asset_path = workspace_root / "assets" / asset_id / "data"
        asset_path.parent.mkdir(parents=True)
        asset_path.write_text("asset content")
        
        # Create resource
        resource = AssetResource(f"asset://{asset_id}", workspace_root)
        
        # Test properties
        assert resource.uri == f"asset://{asset_id}"
        assert resource.asset_id == asset_id
        assert resource.path == asset_path
        assert resource.exists()
        
        # Test read
        content = resource.read()
        assert content == b"asset content"
        
        # Test write
        resource.write(b"new asset content")
        assert resource.read() == b"new asset content"


def test_file_resource_nonexistent():
    """Test FileResource with nonexistent file."""
    resource = FileResource("file:///nonexistent/file.txt")
    assert not resource.exists()
    
    with pytest.raises(FileNotFoundError):
        resource.read()


def test_folder_resource_nonexistent():
    """Test FolderResource with nonexistent directory."""
    resource = FolderResource("file:///nonexistent/dir")
    assert not resource.exists()
    assert resource.list_files() == []
