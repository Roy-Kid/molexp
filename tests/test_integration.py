"""Integration tests for workspace and repositories."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from molexp.workspace import Workspace
from molexp.models import RunStatus, AssetType, WorkflowTemplate
from molexp.id_utils import generate_asset_id


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace.from_path(tmpdir)
        yield workspace


def test_workspace_initialization(temp_workspace):
    """Test workspace initialization."""
    assert temp_workspace.root.exists()
    assert (temp_workspace.root / "projects").exists()
    assert (temp_workspace.root / "assets").exists()


def test_project_crud(temp_workspace):
    """Test project CRUD operations."""
    # Create
    project = temp_workspace.create_project(
        project_id="test-project",
        name="Test Project",
        description="A test",
        owner="researcher",
        tags=["test"],
    )
    
    assert project.project_id == "test-project"
    assert project.name == "Test Project"
    
    # Read
    retrieved = temp_workspace.get_project("test-project")
    assert retrieved is not None
    assert retrieved.project_id == project.project_id
    
    # List
    projects = temp_workspace.list_projects()
    assert len(projects) == 1
    assert projects[0].project_id == "test-project"
    
    # Delete
    temp_workspace.delete_project("test-project")
    assert temp_workspace.get_project("test-project") is None


def test_experiment_crud(temp_workspace):
    """Test experiment CRUD operations."""
    # Create project first
    temp_workspace.create_project(
        project_id="test-project",
        name="Test Project",
    )
    
    # Create experiment
    experiment = temp_workspace.create_experiment(
        project_id="test-project",
        experiment_id="exp-1",
        name="Experiment 1",
        workflow_source="workflow.py",
        parameter_space={"param1": [1, 2, 3]},
    )
    
    assert experiment.experiment_id == "exp-1"
    
    # Read
    retrieved = temp_workspace.get_experiment("test-project", "exp-1")
    assert retrieved is not None
    assert retrieved.experiment_id == "exp-1"
    
    # List
    experiments = temp_workspace.list_experiments("test-project")
    assert len(experiments) == 1


def test_run_crud(temp_workspace):
    """Test run CRUD operations."""
    # Create project and experiment
    temp_workspace.create_project(project_id="test-project", name="Test")
    temp_workspace.create_experiment(
        project_id="test-project",
        experiment_id="exp-1",
        name="Exp 1",
        workflow_source="workflow.py",
    )
    
    # Create run
    run = temp_workspace.create_run(
        project_id="test-project",
        experiment_id="exp-1",
        parameters={"param1": 1},
        workflow_file="workflow.py",
    )
    
    assert run.status == RunStatus.PENDING
    
    # Update run status
    run.status = RunStatus.SUCCEEDED
    run.finished_at = datetime.now()
    updated = temp_workspace.update_run(run)
    assert updated.status == RunStatus.SUCCEEDED
    
    # Read
    retrieved = temp_workspace.get_run("test-project", "exp-1", run.run_id)
    assert retrieved is not None
    assert retrieved.status == RunStatus.SUCCEEDED
    
    # List
    runs = temp_workspace.list_runs("test-project", "exp-1")
    assert len(runs) == 1


def test_asset_storage_and_deduplication(temp_workspace):
    """Test asset storage with content-based deduplication."""
    # Create a test file
    test_file = temp_workspace.root / "test.txt"
    test_file.write_text("test content")
    
    from molexp.models import Asset, AssetFile
    from molexp.id_utils import compute_content_hash
    
    content_hash = compute_content_hash(test_file)
    
    # Store first asset
    asset1_id = generate_asset_id()
    asset1 = Asset(
        asset_id=asset1_id,
        type=AssetType.OTHER,
        format="txt",
        created_at=datetime.now(),
        size_bytes=test_file.stat().st_size,
        content_hash=content_hash,
        files=[AssetFile(path="data/test.txt", size=12, hash=content_hash)],
    )
    
    temp_workspace.store_asset(asset1, test_file)
    
    # Verify storage
    retrieved = temp_workspace.get_asset(asset1_id)
    assert retrieved is not None
    assert retrieved.content_hash == content_hash
    
    # Check deduplication: same hash should be found
    found_id = temp_workspace.find_asset_by_hash(content_hash)
    assert found_id == asset1_id
    
    # Store second asset with same content
    asset2_id = generate_asset_id()
    asset2 = Asset(
        asset_id=asset2_id,
        type=AssetType.OTHER,
        format="txt",
        created_at=datetime.now(),
        size_bytes=test_file.stat().st_size,
        content_hash=content_hash,
        files=[AssetFile(path="data/test.txt", size=12, hash=content_hash)],
    )
    
    temp_workspace.store_asset(asset2, test_file)
    
    # Both should be findable by hash (last one wins in index)
    found_id = temp_workspace.find_asset_by_hash(content_hash)
    assert found_id in [asset1_id, asset2_id]


def test_asset_refs_collection(temp_workspace):
    """Test asset references collection."""
    from molexp.models import AssetRefsCollection, AssetRef
    
    # Create project, experiment, and run
    temp_workspace.create_project(project_id="test-project", name="Test")
    temp_workspace.create_experiment(
        project_id="test-project",
        experiment_id="exp-1",
        name="Exp 1",
        workflow_source="workflow.py",
    )
    run = temp_workspace.create_run(
        project_id="test-project",
        experiment_id="exp-1",
        parameters={},
        workflow_file="workflow.py",
    )
    
    # Get initial refs (should be empty)
    refs = temp_workspace.get_asset_refs("test-project", "exp-1", run.run_id)
    assert refs is not None
    assert len(refs.inputs) == 0
    assert len(refs.outputs) == 0
    
    # Add an output ref
    refs.outputs.append(
        AssetRef(
            asset_id="test-asset-id",
            role="output",
            produced_at=datetime.now(),
        )
    )
    
    temp_workspace.save_asset_refs("test-project", "exp-1", run.run_id, refs)
    
    # Retrieve and verify
    retrieved_refs = temp_workspace.get_asset_refs("test-project", "exp-1", run.run_id)
    assert len(retrieved_refs.outputs) == 1
    assert retrieved_refs.outputs[0].asset_id == "test-asset-id"
