"""``ValidateWorkflowIR`` — fourth stage of the §3 pipeline.

Loads a ``PlanWorkflowIR`` artifact from the store, runs the Phase-3 pure
:func:`validate_workflow_ir` validator, and persists the resulting
:class:`PlanValidationReport` as a new artifact of kind ``validation_report``.

The report is **always persisted** — even on failure — so the audit
trail captures failed validations as first-class artifacts that future
repair stages (Phase 8+) can reference via provenance. Only *after*
persistence does the stage optionally raise
:class:`StagePersistedFailureError` (caught by :class:`StageRunner` and
re-raised as :class:`StageExecutionError`) when ``raise_on_failure`` is
True (the default) and the report failed.

JSON parse / schema errors on the input artifact are themselves treated
as validation failures: we synthesize a one-violation PlanValidationReport
(code ``"ir_parse_error"``) and persist it before raising. This keeps
the always-persist contract even when the agent emits unparseable JSON.

``raise_on_failure`` is keyword-only. Strict mode (the default) is the
common case for pipeline composition; soft mode lets callers branch on
the report's contents without losing the audit artifact.
"""

from __future__ import annotations

import json
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StagePersistedFailureError
from molexp.harness.schemas import (
    PlanArtifactRef,
    PlanValidationReport,
    PlanWorkflowIR,
    ValidationViolation,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.validators.render import render_violations
from molexp.harness.validators.workflow_ir import WorkflowIRValidator

__all__ = ["ValidateWorkflowIR"]


class ValidateWorkflowIR(Stage):
    """Validate a PlanWorkflowIR artifact; persist a PlanValidationReport artifact."""

    name: ClassVar[str] = "validate_workflow_ir"

    def __init__(self, *, raise_on_failure: bool = True) -> None:
        self._raise_on_failure = raise_on_failure

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        ir_ref = require_latest(ctx, "workflow_ir", stage=self.name)
        target_id = ir_ref.id
        raw = ctx.artifact_store.get(target_id)

        # Parse the IR; on any parse/schema error, synthesize a
        # parse-error PlanValidationReport so the audit trail captures it.
        try:
            ir = PlanWorkflowIR.model_validate_json(raw)
        except Exception as exc:
            report = PlanValidationReport.from_violations(
                target_kind="workflow_ir",
                target_id=target_id,
                violations=[
                    ValidationViolation(
                        code="ir_parse_error",
                        message=f"PlanWorkflowIR JSON failed schema validation: {exc!r}",
                        severity="error",
                    )
                ],
            )
            report_ref = ctx.artifact_store.put_json(
                kind="validation_report",
                obj=json.loads(report.model_dump_json()),
                created_by="ValidateWorkflowIR",
                parent_ids=[target_id],
            )
            if self._raise_on_failure:
                raise StagePersistedFailureError(
                    report_ref,
                    f"PlanWorkflowIR parse failed: {exc!r}",
                ) from exc
            return report_ref

        report = WorkflowIRValidator.validate(ir)

        # Persist the report unconditionally — failed validations stay
        # first-class artifacts in the audit trail.
        report_ref = ctx.artifact_store.put_json(
            kind="validation_report",
            obj=json.loads(report.model_dump_json()),
            created_by="ValidateWorkflowIR",
            parent_ids=[target_id],
        )

        if not report.passed and self._raise_on_failure:
            raise StagePersistedFailureError(
                report_ref,
                f"PlanWorkflowIR validation failed:\n{render_violations(report.violations)}",
            )

        return report_ref
