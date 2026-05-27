"""Tests for ValidateWorkflowIR stage (Phase 7).

Locks:
- Stage subclass, name="validate_workflow_ir"
- positional: workflow_ir_artifact_id; keyword-only: raise_on_failure (default True)
- happy path persists validation_report artifact + returns its ref
- strict failure (default) raises StageExecutionError AFTER persisting report
- soft failure (raise_on_failure=False) returns ref without raising
- report is always persisted regardless of pass/fail
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-validate",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
    )


def _valid_workflow_ir_dict() -> dict:
    return {
        "id": "wf-ok",
        "name": "wf",
        "objective": "x",
        "inputs": {},
        "tasks": [
            {
                "id": "t1",
                "name": "T",
                "purpose": "p",
                "task_type": "tt",
                "inputs": {},
                "outputs": {"out": "out.txt"},
            }
        ],
        "edges": [],
        "expected_outputs": [],
    }


def _invalid_workflow_ir_dict_with_cycle() -> dict:
    """Two tasks with edges in both directions → cyclic_dependency violation."""
    return {
        "id": "wf-bad",
        "name": "wf",
        "objective": "x",
        "inputs": {},
        "tasks": [
            {
                "id": "t1",
                "name": "T1",
                "purpose": "p",
                "task_type": "tt",
                "inputs": {},
                "outputs": {"out1": "out1.txt"},
            },
            {
                "id": "t2",
                "name": "T2",
                "purpose": "p",
                "task_type": "tt",
                "inputs": {},
                "outputs": {"out2": "out2.txt"},
            },
        ],
        "edges": [
            {"source_task_id": "t1", "target_task_id": "t2"},
            {"source_task_id": "t2", "target_task_id": "t1"},
        ],
        "expected_outputs": [],
    }


def _seed_workflow_ir(ctx, obj):
    return ctx.artifact_store.put_json(
        kind="workflow_ir", obj=obj, created_by="seed", parent_ids=[]
    )


def test_validate_workflow_ir_name() -> None:
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    assert ValidateWorkflowIR.name == "validate_workflow_ir"


def test_validate_workflow_ir_is_stage_subclass() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    assert issubclass(ValidateWorkflowIR, Stage)


def test_raise_on_failure_is_keyword_only() -> None:
    """Calling ValidateWorkflowIR('x', True) must fail — True is positional, not allowed."""
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    with pytest.raises(TypeError):
        ValidateWorkflowIR("x", True)  # type: ignore[misc]


def test_happy_path_persists_validation_report_and_returns_ref(ctx) -> None:
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    ir_ref = _seed_workflow_ir(ctx, _valid_workflow_ir_dict())
    stage = ValidateWorkflowIR(workflow_ir_artifact_id=ir_ref.id)
    report_ref = asyncio.run(stage.run(ctx))

    assert report_ref.kind == "validation_report"
    assert ir_ref.id in report_ref.parent_ids

    # Loaded report parses as ValidationReport and is passed.
    from molexp.harness.schemas.validation import ValidationReport

    raw = ctx.artifact_store.get(report_ref.id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is True
    assert report.target_kind == "workflow_ir"


def test_strict_failure_raises_after_persisting_report(ctx) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    ir_ref = _seed_workflow_ir(ctx, _invalid_workflow_ir_dict_with_cycle())
    stage = ValidateWorkflowIR(workflow_ir_artifact_id=ir_ref.id)  # default raise_on_failure=True

    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx))
    assert "cyclic_dependency" in str(exc.value)

    # Despite raising, the validation_report MUST be persisted.
    reports = ctx.artifact_store.list_by_kind("validation_report")
    assert len(reports) == 1
    from molexp.harness.schemas.validation import ValidationReport

    raw = ctx.artifact_store.get(reports[0].id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False
    assert any(v.code == "cyclic_dependency" for v in report.violations)


def test_soft_failure_returns_ref_without_raising(ctx) -> None:
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    ir_ref = _seed_workflow_ir(ctx, _invalid_workflow_ir_dict_with_cycle())
    stage = ValidateWorkflowIR(workflow_ir_artifact_id=ir_ref.id, raise_on_failure=False)
    report_ref = asyncio.run(stage.run(ctx))
    assert report_ref.kind == "validation_report"

    from molexp.harness.schemas.validation import ValidationReport

    raw = ctx.artifact_store.get(report_ref.id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False


def test_unparseable_input_persists_parse_error_report_and_raises(ctx) -> None:
    """Garbage JSON for WorkflowIR → parse-error ValidationReport persisted, then raise.

    The always-persist contract MUST hold even when the input is so
    malformed that the validator can't be invoked. The stage synthesizes
    a one-violation ValidationReport (code 'ir_parse_error') and raises
    ``StagePersistedFailureError`` (a ``StageExecutionError`` subclass).
    """
    from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    # Seed a non-WorkflowIR artifact under the kind="workflow_ir" slot.
    bogus_ref = ctx.artifact_store.put_json(
        kind="workflow_ir",
        obj={"not": "a workflow ir at all"},
        created_by="seed",
        parent_ids=[],
    )
    stage = ValidateWorkflowIR(workflow_ir_artifact_id=bogus_ref.id)

    with pytest.raises(StageExecutionError) as exc_info:
        asyncio.run(stage.run(ctx))
    assert isinstance(exc_info.value, StagePersistedFailureError)

    # The parse-error ValidationReport MUST be persisted.
    reports = ctx.artifact_store.list_by_kind("validation_report")
    assert len(reports) == 1
    raw = ctx.artifact_store.get(reports[0].id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False
    assert report.target_kind == "workflow_ir"
    assert any(v.code == "ir_parse_error" for v in report.violations)
    # And the persisted ref is reachable from the exception.
    assert exc_info.value.persisted_ref.id == reports[0].id


def test_unparseable_input_soft_mode_returns_parse_error_report(ctx) -> None:
    """raise_on_failure=False on unparseable IR returns the parse-error report."""
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR

    bogus_ref = ctx.artifact_store.put_json(
        kind="workflow_ir",
        obj={"oops": True},
        created_by="seed",
        parent_ids=[],
    )
    stage = ValidateWorkflowIR(workflow_ir_artifact_id=bogus_ref.id, raise_on_failure=False)
    report_ref = asyncio.run(stage.run(ctx))
    raw = ctx.artifact_store.get(report_ref.id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False
    assert any(v.code == "ir_parse_error" for v in report.violations)
