"""Tests for the ``ValidateTestSpec`` stage (spec ``harness-run-mode-01-substrate``, T03).

Mirrors the canonical validator-stage shape verified in
``ValidateWorkflowSource`` (``stages/validate_workflow_source.py``):

- input resolved via ``require_latest("test_spec")``;
- a ``ValidationReport`` is ALWAYS persisted as a ``"validation_report"``
  artifact whose ``parent_ids`` carry the test_spec artifact id;
- on failure with the default ``raise_on_failure=True`` the stage raises
  ``StagePersistedFailureError`` AFTER persisting, with ``persisted_ref``
  pointing at the failing report;
- ``raise_on_failure=False`` returns the failing report ref without raising;
- when a ``workflow_ir`` artifact exists in the store, the stage cross-checks
  the TestSpec target against the IR (wrapping
  ``validators.validate_test_spec(spec, ir=...)``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from molexp.harness import ArtifactRef
    from molexp.harness.core.run_context import HarnessRunContext

# --------------------------------------------------------------- fixtures


@pytest.fixture()
def ctx(tmp_path: Path) -> HarnessRunContext:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-vts",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _test_spec_dict(
    *,
    target_task_id: str | None = "task-square",
    target_workflow_id: str | None = None,
) -> dict:
    from molexp.harness import TestSpec

    spec = TestSpec(
        id="ts-001",
        name="unit: square",
        kind="unit_test",
        target_task_id=target_task_id,
        target_workflow_id=target_workflow_id,
        description="the square task squares its input",
    )
    return json.loads(spec.model_dump_json())


def _workflow_ir_dict(task_id: str) -> dict:
    from molexp.harness import TaskIR, WorkflowIR

    ir = WorkflowIR(
        id="wf-x",
        name="demo",
        objective="exercise the square task",
        inputs={},
        tasks=[
            TaskIR(
                id=task_id,
                name="square",
                purpose="square the input integers",
                task_type="compute",
                inputs={},
                outputs={"squares": "dataset"},
            )
        ],
        edges=[],
        expected_outputs=[],
    )
    return json.loads(ir.model_dump_json())


def _seed(ctx: HarnessRunContext, kind: str, obj: dict) -> ArtifactRef:
    return ctx.artifact_store.put_json(kind=kind, obj=obj, created_by="seed", parent_ids=[])


# ------------------------------------------------------------ stage shape


def test_stage_name_and_subclass() -> None:
    from molexp.harness import ValidateTestSpec
    from molexp.harness.core.stage import Stage

    assert ValidateTestSpec.name == "validate_test_spec"
    assert issubclass(ValidateTestSpec, Stage)


def test_ctor_raise_on_failure_is_keyword_only() -> None:
    from molexp.harness import ValidateTestSpec

    with pytest.raises(TypeError):
        ValidateTestSpec(False)  # type: ignore[call-arg]


# ------------------------------------------------------------- happy path


def test_happy_path_persists_passing_report(ctx) -> None:
    from molexp.harness import ValidateTestSpec, ValidationReport

    spec_ref = _seed(ctx, "test_spec", _test_spec_dict())
    report_ref = asyncio.run(ValidateTestSpec().run(ctx))

    assert report_ref.kind == "validation_report"
    assert spec_ref.id in report_ref.parent_ids

    raw = ctx.artifact_store.get(report_ref.id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is True
    assert report.target_kind == "test_spec"


class TestValidateTestSpecBundle:
    """ValidateTestSpec validates every member of a TestSpecBundle."""

    def test_bundle_validates_every_member_spec(self, ctx) -> None:
        """ac-004 — a TestSpecBundle is validated per-member into one report."""
        from molexp.harness import ValidateTestSpec, ValidationReport

        bundle = {
            "id": "tsb-1",
            "bound_workflow_id": "wf",
            "specs": [_test_spec_dict(), _test_spec_dict()],
        }
        _seed(ctx, "test_spec", bundle)
        report_ref = asyncio.run(ValidateTestSpec().run(ctx))
        report = ValidationReport.model_validate(json.loads(ctx.artifact_store.get(report_ref.id)))
        assert report.passed is True
        assert report.target_kind == "test_spec"

    def test_empty_bundle_fails_validation(self, ctx) -> None:
        """ac-004 — a bundle with no specs is itself a violation."""
        from molexp.harness import StagePersistedFailureError, ValidateTestSpec, ValidationReport

        _seed(ctx, "test_spec", {"id": "tsb-empty", "bound_workflow_id": "wf", "specs": []})
        with pytest.raises(StagePersistedFailureError):
            asyncio.run(ValidateTestSpec().run(ctx))

        report_ref = ctx.artifact_store.latest_by_kind("validation_report")
        report = ValidationReport.model_validate(json.loads(ctx.artifact_store.get(report_ref.id)))
        assert report.passed is False
        assert any(v.code == "empty_test_spec_bundle" for v in report.violations)


# ----------------------------------------------- workflow_ir cross-check


def test_cross_check_passes_when_target_task_in_ir(ctx) -> None:
    from molexp.harness import ValidateTestSpec, ValidationReport

    _seed(ctx, "workflow_ir", _workflow_ir_dict(task_id="task-square"))
    _seed(ctx, "test_spec", _test_spec_dict(target_task_id="task-square"))
    report_ref = asyncio.run(ValidateTestSpec().run(ctx))

    raw = ctx.artifact_store.get(report_ref.id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is True
    assert report.target_kind == "test_spec"


def test_cross_check_fails_when_target_task_not_in_ir(ctx) -> None:
    from molexp.harness import StagePersistedFailureError, ValidateTestSpec, ValidationReport

    _seed(ctx, "workflow_ir", _workflow_ir_dict(task_id="task-other"))
    _seed(ctx, "test_spec", _test_spec_dict(target_task_id="task-square"))

    with pytest.raises(StagePersistedFailureError) as exc_info:
        asyncio.run(ValidateTestSpec().run(ctx))

    # Report persisted despite the raise (always-persist contract).
    reports = ctx.artifact_store.list_by_kind("validation_report")
    assert len(reports) == 1
    raw = ctx.artifact_store.get(reports[0].id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False
    assert any(v.code == "unknown_task_target" for v in report.violations)
    assert exc_info.value.persisted_ref.id == reports[0].id
    assert exc_info.value.persisted_ref.kind == "validation_report"


# ---------------------------------------------- red path: missing target


def test_missing_target_persists_failing_report_then_raises(ctx) -> None:
    from molexp.harness import StagePersistedFailureError, ValidateTestSpec, ValidationReport

    _seed(ctx, "test_spec", _test_spec_dict(target_task_id=None, target_workflow_id=None))

    with pytest.raises(StagePersistedFailureError) as exc_info:
        asyncio.run(ValidateTestSpec().run(ctx))

    reports = ctx.artifact_store.list_by_kind("validation_report")
    assert len(reports) == 1
    raw = ctx.artifact_store.get(reports[0].id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False
    assert any(v.code == "missing_target" for v in report.violations)
    assert exc_info.value.persisted_ref.id == reports[0].id


def test_missing_target_returns_failing_ref_when_raise_disabled(ctx) -> None:
    from molexp.harness import ValidateTestSpec, ValidationReport

    spec_ref = _seed(
        ctx, "test_spec", _test_spec_dict(target_task_id=None, target_workflow_id=None)
    )
    report_ref = asyncio.run(ValidateTestSpec(raise_on_failure=False).run(ctx))

    assert report_ref.kind == "validation_report"
    assert spec_ref.id in report_ref.parent_ids
    raw = ctx.artifact_store.get(report_ref.id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False
