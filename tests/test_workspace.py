"""Tests for Workspace class with metadata/object separation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from molexp.workspace import Workspace, Project, Experiment, Run


def test_workspace_creation():
    """Test workspace creation with materialize pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create workspace (no side effects)
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        
        # Verify metadata auto-generated
        assert workspace.id == "test-workspace"  # Slugified from name
        assert workspace.name == "Test Workspace"
        assert workspace.metadata.created_at is not None
        assert workspace.root == Path(tmpdir).resolve()
        
        # Explicitly materialize
        workspace.materialize()
        
        # Verify filesystem created
        assert (workspace.root / "workspace.json").exists()
        assert workspace.assets is not None


def test_workspace_load():
    """Test loading workspace from disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and materialize workspace
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        
        # Load workspace from disk
        loaded = Workspace.load(tmpdir)
        
        # Verify metadata preserved
        assert loaded.id == workspace.id
        assert loaded.name == workspace.name
        assert loaded.metadata.created_at == workspace.metadata.created_at


def test_project_creation():
    """Test creating a project with auto-generated metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        
        # Create project (auto-materialized by workspace.create_project)
        project = workspace.create_project(name="Test Project")
        
        # Verify auto-generated fields
        assert project.id == "test-project"  # Slugified from name
        assert project.name == "Test Project"
        assert project.metadata.created_at is not None
        assert project.metadata.updated_at is not None
        
        # Verify filesystem created
        project_dir = workspace.root / "projects" / project.id
        assert project_dir.exists()
        assert (project_dir / "project.json").exists()
        
        # Verify project has asset library
        assert project.assets is not None


def test_project_get_and_list():
    """Test retrieving and listing projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        
        # Create multiple projects
        project1 = workspace.create_project(name="Project 1")
        project2 = workspace.create_project(name="Project 2")
        project3 = workspace.create_project(name="Project 3")
        
        # Get by ID
        retrieved = workspace.get_project("project-1")
        assert retrieved.id == project1.id
        assert retrieved.name == project1.name
        
        # Get by name (slugified)
        retrieved2 = workspace.get_project("Project 2")
        assert retrieved2.id == project2.id
        
        # List projects
        projects = workspace.list_projects()
        assert len(projects) == 3
        assert {p.name for p in projects} == {"Project 1", "Project 2", "Project 3"}


def test_experiment_creation():
    """Test creating an experiment with auto-generated UUID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        project = workspace.create_project(name="Test Project")
        
        # Create experiment (auto-materialized by project.create_experiment)
        experiment = project.create_experiment(name="Test Experiment")
        
        # Verify auto-generated fields
        assert experiment.id is not None  # UUID auto-generated
        assert experiment.name == "Test Experiment"
        assert experiment.metadata.created_at is not None
        
        # Verify parent references
        assert experiment.project is project
        assert experiment.workspace is workspace
        
        # Verify filesystem created
        exp_dir = workspace.root / "projects" / project.id / "experiments" / experiment.id
        assert exp_dir.exists()
        assert (exp_dir / "experiment.json").exists()
        
        # Verify experiment has asset library
        assert experiment.assets is not None


def test_run_creation():
    """Test creating a run with auto-generated UUID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        project = workspace.create_project(name="Test Project")
        experiment = project.create_experiment(name="Test Experiment")
        
        # Create run (auto-materialized by experiment.create_run)
        run = experiment.create_run(parameters={"lr": 0.001, "batch_size": 32})
        
        # Verify auto-generated fields
        assert run.id is not None  # UUID auto-generated
        assert run.metadata.created_at is not None
        assert run.parameters == {"lr": 0.001, "batch_size": 32}
        assert run.status == "pending"
        
        # Verify parent reference
        assert run.experiment is experiment
        
        # Verify filesystem created
        run_dir = (
            workspace.root / "projects" / project.id / 
            "experiments" / experiment.id / "runs" / run.id
        )
        assert run_dir.exists()
        assert (run_dir / "run.json").exists()


def test_hierarchical_asset_libraries():
    """Test that asset libraries are isolated at each level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        project = workspace.create_project(name="Test Project")
        experiment = project.create_experiment(name="Test Experiment")
        run = experiment.create_run(parameters={})
        
        # Create test file
        source_file = Path(tmpdir) / "test.txt"
        source_file.write_text("test data")
        
        # Import asset at each level (using correct API)
        ws_asset = workspace.assets.import_asset("ws_asset", source_file)
        proj_asset = project.assets.import_asset("proj_asset", source_file)
        exp_asset = experiment.assets.import_asset("exp_asset", source_file)
        
        # Verify isolation - each library only has its own asset
        assert len(workspace.assets.list_assets()) == 1
        assert workspace.assets.list_assets()[0].name == "ws_asset"
        
        assert len(project.assets.list_assets()) == 1
        assert project.assets.list_assets()[0].name == "proj_asset"
        
        assert len(experiment.assets.list_assets()) == 1
        assert experiment.assets.list_assets()[0].name == "exp_asset"


def test_run_context_creation():
    """Test creating runtime context from run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        project = workspace.create_project(name="Test Project")
        experiment = project.create_experiment(name="Test Experiment")
        run = experiment.create_run(parameters={"lr": 0.001})

        # Create context via context manager
        run_ctx = run.context()
        assert run_ctx.run is run

        # Verify context has correct IDs and work_dir
        ctx = run_ctx.context
        assert ctx.run_id == run.id
        assert ctx.experiment_id == experiment.id
        assert ctx.project_id == project.id
        assert run_ctx.work_dir == (
            workspace.root / "projects" / project.id /
            "experiments" / experiment.id / "runs" / run.id
        )


def test_metadata_persistence():
    """Test that metadata persists correctly across save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create workspace and project
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        workspace.materialize()
        project = workspace.create_project(name="Test Project")
        
        # Reload workspace
        workspace2 = Workspace.load(tmpdir)
        project2 = workspace2.get_project(project.id)
        
        # Verify metadata matches
        assert project2.id == project.id
        assert project2.name == project.name
        assert project2.metadata.created_at == project.metadata.created_at


def test_no_side_effects_on_construction():
    """Test that constructing entities has no filesystem side effects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create workspace without materializing
        workspace = Workspace(root=tmpdir, name="Test Workspace")
        
        # Verify no files created yet
        assert not (Path(tmpdir) / "workspace.json").exists()
        
        # Now materialize
        workspace.materialize()
        
        # Verify files created
        assert (Path(tmpdir) / "workspace.json").exists()
