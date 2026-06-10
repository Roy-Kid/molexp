"""Tests for ValidateBoundWorkflow stage (Phase 8)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def _valid_ir_dict() -> dict:
    return {
        "id": "wf-ok",
        "name": "wf",
        "objective": "x",
        "inputs": {"n": {"value": 1, "source": "user_provided"}},
        "tasks": [
            {
                "id": "t1",
                "name": "T",
                "purpose": "p",
                "task_type": "tt",
                "inputs": {"n": {"value": 1, "source": "user_provided"}},
                "outputs": {"out": "out.txt"},
            }
        ],
        "edges": [],
        "expected_outputs": [],
    }


def _valid_bw_dict() -> dict:
    return {
        "id": "bw-ok",
        "workflow_ir_id": "wf-ok",
        "tasks": [
            {
                "id": "b1",
                "ir_task_id": "t1",
                "capability_id": "cap.x",
                "package": "pkg",
                "callable": "pkg.X",
                "parameters": {"n": {"value": 1, "source": "user_provided"}},
                "inputs": {"n": "x"},
                "outputs": {"out": "out.txt"},
            }
        ],
        "edges": [],
        "execution_backend": "local",
        "environment": {},
        "resource_policy": {
            "backend": "local",
            "max_runtime_s": 3600,
            "denied_paths": ["/", "~/.ssh"],
        },
    }


def _bw_with_bad_capability_dict() -> dict:
    d = _valid_bw_dict()
    d["tasks"][0]["capability_id"] = "ghost.capability"
    return d


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-vbw",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _seed_pair(ctx, ir_dict, bw_dict):
    ir_ref = ctx.artifact_store.put_json(
        kind="workflow_ir", obj=ir_dict, created_by="seed", parent_ids=[]
    )
    bw_ref = ctx.artifact_store.put_json(
        kind="bound_workflow", obj=bw_dict, created_by="seed", parent_ids=[ir_ref.id]
    )
    return ir_ref, bw_ref


def test_name_and_stage_subclass() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow

    assert ValidateBoundWorkflow.name == "validate_bound_workflow"
    assert issubclass(ValidateBoundWorkflow, Stage)


def test_raise_on_failure_is_keyword_only() -> None:
    from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow

    with pytest.raises(TypeError):
        ValidateBoundWorkflow("bw", "ir", True)  # type: ignore[misc]


def test_happy_path_persists_report_and_returns_ref(ctx) -> None:
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow

    ir_ref, bw_ref = _seed_pair(ctx, _valid_ir_dict(), _valid_bw_dict())
    stage = ValidateBoundWorkflow()
    report_ref = asyncio.run(stage.run(ctx))
    assert report_ref.kind == "validation_report"
    assert bw_ref.id in report_ref.parent_ids
    assert ir_ref.id in report_ref.parent_ids

    report = ValidationReport.model_validate(json.loads(ctx.artifact_store.get(report_ref.id)))
    assert report.passed is True


def test_strict_failure_raises_after_persisting(ctx) -> None:
    """Use missing baseline deny path to trigger a structural error."""
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow

    bad_bw = _valid_bw_dict()
    bad_bw["resource_policy"]["denied_paths"] = []  # missing baseline deny
    _seed_pair(ctx, _valid_ir_dict(), bad_bw)

    stage = ValidateBoundWorkflow()
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx))
    assert "missing_baseline_deny" in str(exc.value)

    reports = ctx.artifact_store.list_by_kind("validation_report")
    assert len(reports) == 1
    report = ValidationReport.model_validate(json.loads(ctx.artifact_store.get(reports[0].id)))
    assert report.passed is False


def test_soft_failure_returns_ref(ctx) -> None:
    from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow

    bad_bw = _valid_bw_dict()
    bad_bw["resource_policy"]["denied_paths"] = []
    _seed_pair(ctx, _valid_ir_dict(), bad_bw)

    stage = ValidateBoundWorkflow(raise_on_failure=False)
    ref = asyncio.run(stage.run(ctx))
    assert ref.kind == "validation_report"


def test_uses_ctx_capability_registry_when_set(tmp_path: Path) -> None:
    """When registry has no matching capability_id, unknown_capability fires."""
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry
    from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    # Empty registry → cap.x is unknown
    registry = InMemoryCapabilityRegistry()
    ctx = HarnessRunContext(
        run_id="run-vbw-reg",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
        capability_registry=registry,
    )
    _seed_pair(ctx, _valid_ir_dict(), _valid_bw_dict())
    stage = ValidateBoundWorkflow()
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx))
    assert "unknown_capability" in str(exc.value)
