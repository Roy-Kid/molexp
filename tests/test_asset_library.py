"""Tests for asset library and workflows."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from molexp.workspace.asset import Asset, AssetLibrary, AssetWorkflow
from molexp.workspace.resource import FileResource


def test_asset_library_create_asset():
    """Test creating asset from path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_root = Path(tmpdir) / "library"
        source_file = Path(tmpdir) / "source.txt"
        source_file.write_text("test data")
        
        library = AssetLibrary(library_root)
        
        # Create asset
        asset = library.create_asset("test_asset", source_file)
        
        # Verify asset properties
        assert asset.name == "test_asset"
        assert asset.asset_id is not None  # Auto-generated
        assert asset.created_at is not None  # Auto-generated
        assert asset.uri == f"asset://{asset.asset_id}"
        
        # Verify asset was stored
        assert (library_root / asset.asset_id / "data").exists()
        
        # Verify we can retrieve it
        retrieved = library.get_asset("test_asset")
        assert retrieved.asset_id == asset.asset_id
        assert retrieved.name == "test_asset"


def test_asset_library_add_asset():
    """Test adding asset from resource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_root = Path(tmpdir) / "library"
        source_file = Path(tmpdir) / "source.txt"
        source_file.write_text("test data")
        
        library = AssetLibrary(library_root)
        resource = FileResource(f"file://{source_file}")
        
        # Add asset
        asset = library.add_asset("test_asset", resource)
        
        # Verify asset properties
        assert asset.name == "test_asset"
        assert asset.asset_id is not None  # Auto-generated
        assert asset.created_at is not None  # Auto-generated
        
        # Verify we can retrieve it
        retrieved = library.get_asset("test_asset")
        assert retrieved.asset_id == asset.asset_id


def test_asset_library_duplicate_name():
    """Test that duplicate asset names raise error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_root = Path(tmpdir) / "library"
        source_file = Path(tmpdir) / "source.txt"
        source_file.write_text("test data")
        
        library = AssetLibrary(library_root)
        
        # Create first asset
        library.create_asset("test_asset", source_file)
        
        # Try to create duplicate
        with pytest.raises(ValueError, match="already exists"):
            library.create_asset("test_asset", source_file)


def test_asset_library_list_assets():
    """Test listing all assets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_root = Path(tmpdir) / "library"
        
        library = AssetLibrary(library_root)
        
        # Create multiple assets
        for i in range(3):
            source_file = Path(tmpdir) / f"source{i}.txt"
            source_file.write_text(f"data {i}")
            library.create_asset(f"asset_{i}", source_file)
        
        # List assets
        assets = library.list_assets()
        assert len(assets) == 3
        assert {a.name for a in assets} == {"asset_0", "asset_1", "asset_2"}


def test_asset_workflow():
    """Test asset workflow execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_root = Path(tmpdir) / "library"
        library = AssetLibrary(library_root)
        
        # Define workflow steps
        def step1(**kwargs):
            # Create a file
            file_path = Path(tmpdir) / "workflow_output.txt"
            file_path.write_text("workflow data")
            return {"asset_path": str(file_path)}
        
        def step2(**kwargs):
            # Add asset name
            return {"asset_name": "workflow_asset"}
        
        # Create and register workflow
        workflow = AssetWorkflow("test_workflow", [step1, step2])
        library.add_workflow("test_workflow", workflow)
        
        # Execute workflow
        asset = library.run_workflow("test_workflow")
        
        # Verify asset was created
        assert asset.name == "workflow_asset"
        assert asset.asset_id is not None
        
        # Verify we can retrieve it
        retrieved = library.get_asset("workflow_asset")
        assert retrieved.asset_id == asset.asset_id


def test_asset_workflow_missing_keys():
    """Test workflow fails if final step doesn't return required keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_root = Path(tmpdir) / "library"
        library = AssetLibrary(library_root)
        
        # Define workflow that doesn't return required keys
        def bad_step(**kwargs):
            return {"some_key": "value"}
        
        workflow = AssetWorkflow("bad_workflow", [bad_step])
        library.add_workflow("bad_workflow", workflow)
        
        # Execute workflow should fail
        with pytest.raises(ValueError, match="must return 'asset_name' and 'asset_path'"):
            library.run_workflow("bad_workflow")
