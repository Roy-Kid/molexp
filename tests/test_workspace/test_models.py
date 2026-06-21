"""Tests for workspace.models — frozen metadata."""

from datetime import datetime

import pytest

from molexp.workspace.models import (
    ErrorInfo,
    ExperimentMetadata,
    ProjectMetadata,
    RunMetadata,
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
        with pytest.raises(Exception):  # noqa: B017
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
        with pytest.raises(Exception):  # noqa: B017
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
    def test_workflow_snapshot_is_opaque_dict(self):
        # workflow_snapshot is now an opaque JSON dict — workspace
        # never imports the canonical type. Workflow-layer callers
        # may dump their pydantic ``WorkflowSnapshotRef`` to a dict
        # before passing it in.
        snap = {"source": "train.py", "git_commit": "abc"}
        m = RunMetadata(id="run-1", workflow_snapshot=snap)
        assert m.workflow_snapshot is not None
        assert m.workflow_snapshot["source"] == "train.py"

    def test_error_info(self):
        err = ErrorInfo(type="ValueError", message="bad", timestamp=datetime.now())
        m = RunMetadata(id="run-1", error=err)
        assert m.error.type == "ValueError"

    def test_hot_state_keys_relocated_to_ops(self):
        # status / finished_at / execution_history / labels moved to the OKF
        # _ops/run.json sidecar (wsokf-10); a legacy run.json carrying them
        # loads (extra="ignore") but exposes none.
        m = RunMetadata.model_validate(
            {"id": "run-1", "status": "running", "finished_at": None, "labels": {}}
        )
        assert m.id == "run-1"
        for removed in ("status", "finished_at", "execution_history", "labels"):
            assert not hasattr(m, removed)

    def test_legacy_last_step_key_ignored(self):
        # Old run.json files written before the walltime-chunking removal
        # carry a top-level "last_step" key. RunMetadata must keep pydantic's
        # default extra="ignore" so such legacy data loads without error and
        # exposes no last_step attribute.
        m = RunMetadata.model_validate({"id": "run-1", "last_step": 7})
        assert m.id == "run-1"
        assert not hasattr(m, "last_step")
