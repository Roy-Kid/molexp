"""Tests for core models."""

from datetime import datetime

import pytest

from molexp.models import (Asset, AssetRef, AssetType, Experiment, Project,
                           Run, RunStatus, WorkflowSnapshot, WorkflowTemplate)


def test_project_validation():
    """Test project model validation."""
    project = Project(
        id="test-project",
        name="Test Project",
        description="A test project",
        owner="researcher",
        tags=["test", "demo"],
        created_at=datetime.now(),
    )

    assert project.id == "test-project"
    assert project.name == "Test Project"
    assert project.path == "projects/test-project"


def test_id_validation():
    """Test id validation rules."""
    # Valid IDs
    Project(id="valid-project", name="Valid", created_at=datetime.now())
    Project(id="project123", name="Valid", created_at=datetime.now())

    # Invalid IDs
    with pytest.raises(ValueError):
        Project(id="AB", name="Too short", created_at=datetime.now())

    with pytest.raises(ValueError):
        Project(
            id="Invalid_Project", name="Underscore", created_at=datetime.now()
        )

    with pytest.raises(ValueError):
        Project(id="Invalid Project", name="Space", created_at=datetime.now())


def test_experiment_model():
    """Test experiment model."""
    experiment = Experiment(
        id="exp-1",
        id="test-project",
        name="Experiment 1",
        created_at=datetime.now(),
        workflow_template=WorkflowTemplate(source="workflow.py"),
        parameter_space={"param1": [1, 2, 3]},
    )

    assert experiment.id == "exp-1"
    assert experiment.path == "projects/test-project/experiments/exp-1"


def test_run_model():
    """Test run model."""
    run = Run(
        id="20251129_183000_a3b2",
        id="test-project",
        id="exp-1",
        created_at=datetime.now(),
        status=RunStatus.PENDING,
        parameters={"param1": 1},
        workflow_snapshot=WorkflowSnapshot(workflow_file="workflow.py"),
        working_dir="projects/test-project/experiments/exp-1/runs/20251129_183000_a3b2",
    )

    assert run.id == "20251129_183000_a3b2"
    assert run.status == RunStatus.PENDING
    assert (
        run.path == "projects/test-project/experiments/exp-1/runs/20251129_183000_a3b2"
    )


def test_asset_model():
    """Test asset model."""
    asset = Asset(
        asset_id="a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b",
        type=AssetType.STRUCTURE,
        format="pdb",
        created_at=datetime.now(),
        size_bytes=1024,
        content_hash="sha256:abc123",
        files=[],
    )

    assert asset.asset_id == "a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b"
    assert asset.type == AssetType.STRUCTURE
    assert asset.path == "assets/a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b"


def test_asset_ref_model():
    """Test asset reference model."""
    ref = AssetRef(
        asset_id="a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b",
        role="input_structure",
        producer_id="20251129_183000_a3b2",
    )

    assert ref.asset_id == "a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b"
    assert ref.role == "input_structure"
