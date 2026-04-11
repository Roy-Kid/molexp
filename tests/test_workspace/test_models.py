"""Tests for workspace.models — frozen metadata."""

from datetime import datetime

import pytest

from molexp.workspace.models import (
    ErrorInfo,
    ExperimentMetadata,
    ProjectMetadata,
    RunMetadata,
    WorkflowSnapshotRef,
    WorkspaceMetadata,
)


class TestWorkspaceMetadata:
    def test_creation(self):
        m = WorkspaceMetadata(id="lab", name="My Lab")
        assert m.id == "lab"
        assert m.name == "My Lab"
        assert isinstance(m.created_at, datetime)

    def test_frozen(self):
        m = WorkspaceMetadata(id="lab", name="My Lab")
        with pytest.raises(Exception):
            m.name = "changed"

    def test_no_projects_field(self):
        m = WorkspaceMetadata(id="lab", name="My Lab")
        assert not hasattr(m, "projects")

    def test_no_updated_at_field(self):
        m = WorkspaceMetadata(id="lab", name="My Lab")
        assert not hasattr(m, "updated_at")


class TestProjectMetadata:
    def test_creation(self):
        m = ProjectMetadata(id="qm9", name="QM9")
        assert m.description == ""
        assert m.tags == []

    def test_frozen(self):
        m = ProjectMetadata(id="qm9", name="QM9")
        with pytest.raises(Exception):
            m.description = "changed"

    def test_no_experiments_field(self):
        m = ProjectMetadata(id="qm9", name="QM9")
        assert not hasattr(m, "experiments")

    def test_no_assets_field(self):
        m = ProjectMetadata(id="qm9", name="QM9")
        assert not hasattr(m, "assets")

    def test_model_copy_update(self):
        m = ProjectMetadata(id="qm9", name="QM9")
        m2 = m.model_copy(update={"description": "updated"})
        assert m.description == ""
        assert m2.description == "updated"


class TestExperimentMetadata:
    def test_workflow_fields(self):
        m = ExperimentMetadata(
            id="exp-1",
            name="baseline",
            workflow_source="train.py",
            workflow_type="python",
            parameter_space={"lr": [1e-4]},
            git_commit="abc123",
        )
        assert m.workflow_source == "train.py"
        assert m.workflow_type == "python"
        assert m.parameter_space == {"lr": [1e-4]}
        assert m.git_commit == "abc123"

    def test_workflow_fields_optional(self):
        m = ExperimentMetadata(id="exp-1", name="bare")
        assert m.workflow_source is None
        assert m.parameter_space == {}


class TestRunMetadata:
    def test_workflow_snapshot(self):
        snap = WorkflowSnapshotRef(source="train.py", git_commit="abc")
        m = RunMetadata(id="run-1", workflow_snapshot=snap)
        assert m.workflow_snapshot.source == "train.py"
        assert m.status == "pending"

    def test_error_info(self):
        err = ErrorInfo(type="ValueError", message="bad", timestamp=datetime.now())
        m = RunMetadata(id="run-1", error=err)
        assert m.error.type == "ValueError"

    def test_frozen_status_update_via_copy(self):
        m = RunMetadata(id="run-1")
        m2 = m.model_copy(update={"status": "running"})
        assert m.status == "pending"
        assert m2.status == "running"
