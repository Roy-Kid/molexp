"""Tests for the ``ValidateTestSpec`` stage (``stages/validate_test_spec.py``).

Stage wiring, mirroring the canonical ``ValidateWorkflowSource`` shape:

- a ``PlanValidationReport`` is ALWAYS persisted as a ``"validation_report"``
  artifact whose ``parent_ids`` carry the test_spec artifact id;
- on failure (default ``raise_on_failure=True``) the stage raises
  ``StagePersistedFailureError`` AFTER persisting, and the raised message
  surfaces the violation text (dedup'd), not just codes;
- ``raise_on_failure=False`` returns the failing ref without raising;
- a ``workflow_ir`` artifact, when present, cross-checks the TestSpec target;
- every member of a ``TestSpecBundle`` is validated (an empty bundle is itself
  a violation), and a bare ``TestSpec`` is accepted as a one-element bundle.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.schemas import PlanArtifactRef

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
    from molexp.harness.schemas import TestSpec

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
    from molexp.harness.schemas import PlanTaskIR, PlanWorkflowIR

    ir = PlanWorkflowIR(
        id="wf-x",
        name="demo",
        objective="exercise the square task",
        inputs={},
        tasks=[
            PlanTaskIR(
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


def _seed(ctx: HarnessRunContext, kind: str, obj: dict) -> PlanArtifactRef:
    return ctx.artifact_store.put_json(kind=kind, obj=obj, created_by="seed", parent_ids=[])


class TestValidateTestSpec:
    def test_happy_path_persists_passing_report(self, ctx) -> None:
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSpec

        spec_ref = _seed(ctx, "test_spec", _test_spec_dict())
        report_ref = asyncio.run(ValidateTestSpec().run(ctx))

        assert report_ref.kind == "validation_report"
        assert spec_ref.id in report_ref.parent_ids

        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(report_ref.id))
        )
        assert report.passed is True
        assert report.target_kind == "test_spec"

    def test_cross_check_against_ir_fails_and_error_surfaces_message(self, ctx) -> None:
        """A target task absent from the workflow_ir raises with the violation
        MESSAGE (unknown id + known candidates), not just the bare code."""
        from molexp.harness.errors import StagePersistedFailureError
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSpec

        _seed(ctx, "workflow_ir", _workflow_ir_dict(task_id="task-other"))
        _seed(ctx, "test_spec", _test_spec_dict(target_task_id="task-square"))

        with pytest.raises(StagePersistedFailureError) as exc_info:
            asyncio.run(ValidateTestSpec().run(ctx))

        # Report persisted despite the raise (always-persist contract).
        reports = ctx.artifact_store.list_by_kind("validation_report")
        assert len(reports) == 1
        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(reports[0].id))
        )
        assert report.passed is False
        assert any(v.code == "unknown_task_target" for v in report.violations)
        assert exc_info.value.persisted_ref.id == reports[0].id
        assert exc_info.value.persisted_ref.kind == "validation_report"

        rendered = str(exc_info.value)
        assert "unknown_task_target" in rendered
        assert "'task-square'" in rendered  # the unknown target id
        assert "task-other" in rendered  # the known candidate from the IR

    def test_error_deduplicates_repeated_violation_messages(self, ctx) -> None:
        """Identical violations across bundle members render once in the error."""
        from molexp.harness.errors import StagePersistedFailureError
        from molexp.harness.stages import ValidateTestSpec

        _seed(ctx, "workflow_ir", _workflow_ir_dict(task_id="task-other"))
        bundle = {
            "id": "tsb-dup",
            "bound_workflow_id": "wf",
            "specs": [
                _test_spec_dict(target_task_id="task-square"),
                _test_spec_dict(target_task_id="task-square"),
            ],
        }
        _seed(ctx, "test_spec", bundle)

        with pytest.raises(StagePersistedFailureError) as exc_info:
            asyncio.run(ValidateTestSpec().run(ctx))

        assert str(exc_info.value).count("targets unknown task") == 1

    def test_returns_failing_ref_when_raise_disabled(self, ctx) -> None:
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSpec

        spec_ref = _seed(
            ctx, "test_spec", _test_spec_dict(target_task_id=None, target_workflow_id=None)
        )
        report_ref = asyncio.run(ValidateTestSpec(raise_on_failure=False).run(ctx))

        assert report_ref.kind == "validation_report"
        assert spec_ref.id in report_ref.parent_ids
        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(report_ref.id))
        )
        assert report.passed is False

    def test_bundle_validates_every_member_spec(self, ctx) -> None:
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSpec

        bundle = {
            "id": "tsb-1",
            "bound_workflow_id": "wf",
            "specs": [_test_spec_dict(), _test_spec_dict()],
        }
        _seed(ctx, "test_spec", bundle)
        report_ref = asyncio.run(ValidateTestSpec().run(ctx))
        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(report_ref.id))
        )
        assert report.passed is True
        assert report.target_kind == "test_spec"

    def test_empty_bundle_is_itself_a_violation(self, ctx) -> None:
        from molexp.harness.errors import StagePersistedFailureError
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSpec

        _seed(ctx, "test_spec", {"id": "tsb-empty", "bound_workflow_id": "wf", "specs": []})
        with pytest.raises(StagePersistedFailureError):
            asyncio.run(ValidateTestSpec().run(ctx))

        report_ref = ctx.artifact_store.latest_by_kind("validation_report")
        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(report_ref.id))
        )
        assert report.passed is False
        assert any(v.code == "empty_test_spec_bundle" for v in report.violations)
