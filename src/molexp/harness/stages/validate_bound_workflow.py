"""``ValidateBoundWorkflow`` — sixth stage of the §3 pipeline.

Loads BoundWorkflow + WorkflowIR from the store, runs the Phase-3+Phase-4
validator (capability-aware when ``ctx.capability_registry`` is set,
structural-only otherwise), persists the resulting ValidationReport,
optionally raises on failure.

Mirror of Phase-7's :class:`ValidateWorkflowIR` pattern: always-persist
the report — even when JSON parsing of an input fails — then raise
:class:`StagePersistedFailureError` so the runner records the failure
artifact's lineage before the stage error bubbles up.
"""

from __future__ import annotations

import json
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StagePersistedFailureError
from molexp.harness.schemas import (
    ArtifactRef,
    BoundWorkflow,
    ValidationReport,
    ValidationViolation,
    WorkflowIR,
)
from molexp.harness.validators.bound_workflow import validate_bound_workflow

__all__ = ["ValidateBoundWorkflow"]


class ValidateBoundWorkflow(Stage):
    """Validate a BoundWorkflow against its IR; persist ValidationReport."""

    name: ClassVar[str] = "validate_bound_workflow"

    def __init__(
        self,
        bound_workflow_artifact_id: str,
        workflow_ir_artifact_id: str,
        *,
        raise_on_failure: bool = True,
    ) -> None:
        self._bw_artifact_id = bound_workflow_artifact_id
        self._ir_artifact_id = workflow_ir_artifact_id
        self._raise_on_failure = raise_on_failure

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        bw_raw = ctx.artifact_store.get(self._bw_artifact_id)
        ir_raw = ctx.artifact_store.get(self._ir_artifact_id)

        try:
            bw = BoundWorkflow.model_validate_json(bw_raw)
            ir = WorkflowIR.model_validate_json(ir_raw)
        except Exception as exc:
            report = ValidationReport.from_violations(
                target_kind="bound_workflow",
                target_id=self._bw_artifact_id,
                violations=[
                    ValidationViolation(
                        code="bound_workflow_parse_error",
                        message=(
                            f"BoundWorkflow or WorkflowIR JSON failed schema validation: {exc!r}"
                        ),
                        severity="error",
                    )
                ],
            )
            report_ref = ctx.artifact_store.put_json(
                kind="validation_report",
                obj=json.loads(report.model_dump_json()),
                created_by="ValidateBoundWorkflow",
                parent_ids=[self._bw_artifact_id, self._ir_artifact_id],
            )
            if self._raise_on_failure:
                raise StagePersistedFailureError(
                    report_ref,
                    f"BoundWorkflow parse failed: {exc!r}",
                ) from exc
            return report_ref

        report = validate_bound_workflow(
            bw,
            ir,
            workspace_root=ctx.workspace_root,
            registry=ctx.capability_registry,
        )

        report_ref = ctx.artifact_store.put_json(
            kind="validation_report",
            obj=json.loads(report.model_dump_json()),
            created_by="ValidateBoundWorkflow",
            parent_ids=[self._bw_artifact_id, self._ir_artifact_id],
        )

        if not report.passed and self._raise_on_failure:
            error_codes = [v.code for v in report.violations if v.severity == "error"]
            raise StagePersistedFailureError(
                report_ref,
                f"BoundWorkflow validation failed with violations: {error_codes}",
            )

        return report_ref
