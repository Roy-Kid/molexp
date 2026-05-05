"""Tests for workflow versioning (spec: core-versioning).

Covers acceptance criteria:
- ac-001: Workflow(version=...) round-trips through WorkflowVersion JSON
- ac-002: Same workflow_id with conflicting version raises
- ac-009: RunMetadata.workflow_version persisted on first execution
"""

from __future__ import annotations

import json

import pytest

from molexp.workflow import Workflow
from molexp.workflow.version import (
    WorkflowVersion,
    WorkflowVersionConflictError,
)
from molexp.workspace import Workspace


def _make_two_task_workflow(version: str = "1.0.0") -> Workflow:
    wf = Workflow(name="pipeline", version=version)

    @wf.task
    async def fetch(ctx):
        return 1

    @wf.task(depends_on=["fetch"])
    async def transform(ctx):
        return 2

    return wf


class TestWorkflowVersionRoundTrip:
    def test_register_writes_version_record(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Lab")
        wf = _make_two_task_workflow(version="1.0.0")
        spec = wf.build()

        spec.register(ws)

        version_path = ws.root / ".versions" / "workflows" / f"{spec.workflow_id}.json"
        assert version_path.exists()

        with open(version_path) as fh:
            data = json.load(fh)
        record = WorkflowVersion(**data)

        assert record.workflow_id == spec.workflow_id
        assert record.version == "1.0.0"
        assert record.name == "pipeline"
        assert len(record.topology) == 2
        assert record.topology[0].name == "fetch"
        assert record.topology[1].name == "transform"
        assert record.topology[1].depends_on == ("fetch",)

    def test_register_idempotent_for_same_version(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Lab")
        wf = _make_two_task_workflow(version="1.0.0")
        spec = wf.build()

        spec.register(ws)
        version_path = ws.root / ".versions" / "workflows" / f"{spec.workflow_id}.json"
        first_mtime = version_path.stat().st_mtime_ns

        spec.register(ws)
        second_mtime = version_path.stat().st_mtime_ns
        assert first_mtime == second_mtime, "second register must be a no-op (mtime unchanged)"


class TestWorkflowVersionConflict:
    def test_same_workflow_id_different_version_raises(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Lab")
        spec_v1 = _make_two_task_workflow(version="1.0.0").build()
        spec_v2 = _make_two_task_workflow(version="2.0.0").build()

        assert spec_v1.workflow_id == spec_v2.workflow_id  # same topology

        spec_v1.register(ws)
        with pytest.raises(WorkflowVersionConflictError):
            spec_v2.register(ws)


class TestRunRecordsWorkflowVersion:
    def test_run_metadata_carries_workflow_version(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Lab")
        wf = _make_two_task_workflow(version="1.0.0")
        spec = wf.build()

        run = ws.project("p").experiment("e").run()
        with run.start() as ctx:
            ctx.bind_workflow_version(spec)

        version_path = ws.root / ".versions" / "workflows" / f"{spec.workflow_id}.json"
        assert version_path.exists()

        # Re-load run from disk and verify workflow_version was persisted.
        from molexp.workspace.base import _load_metadata
        from molexp.workspace.models import RunMetadata

        meta = _load_metadata(RunMetadata, run.run_dir / "run.json")
        assert meta.workflow_version == "1.0.0"
        assert meta.workflow_id == spec.workflow_id
