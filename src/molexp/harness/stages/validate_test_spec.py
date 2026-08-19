"""``ValidateTestSpec`` — structural gate on the generated TestSpecBundle.

Loads the latest ``test_spec`` artifact (a :class:`TestSpecBundle` carrying
one :class:`TestSpec` per ``BoundTask``), runs the pure
:func:`validate_test_spec` validator over **every** member spec
(cross-checked against the run's ``workflow_ir`` artifact when one exists,
shallow otherwise), merges the violations into one
:class:`PlanValidationReport`, and persists it **always** — on failure the stage
raises :class:`StagePersistedFailureError` after persisting (mirroring
:class:`ValidateWorkflowSource`). An empty bundle (no specs) is itself a
violation.
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
    TestSpecBundle,
    ValidationViolation,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.validators.render import render_violations
from molexp.harness.validators.test_spec import TestSpecValidator

__all__ = ["ValidateTestSpec"]


class ValidateTestSpec(Stage):
    """Validate the generated TestSpec artifact; persist a report."""

    name: ClassVar[str] = "validate_test_spec"

    def __init__(self, *, raise_on_failure: bool = True) -> None:
        self._raise_on_failure = raise_on_failure

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        target = require_latest(ctx, "test_spec", stage=self.name).id
        raw = ctx.artifact_store.get(target)

        try:
            bundle = TestSpecBundle.from_artifact(raw)
        except Exception as exc:
            report = PlanValidationReport.from_violations(
                target_kind="test_spec",
                target_id=target,
                violations=[
                    ValidationViolation(
                        code="test_spec_parse_error",
                        message=f"TestSpec JSON failed schema validation: {exc!r}",
                        severity="error",
                    )
                ],
            )
            return self._persist_and_maybe_raise(
                ctx, report, f"TestSpec parse failed: {exc!r}", target=target
            )

        ir = self._load_ir(ctx)
        violations: list[ValidationViolation] = []
        if not bundle.specs:
            violations.append(
                ValidationViolation(
                    code="empty_test_spec_bundle",
                    message="TestSpecBundle carries no specs; a bound workflow "
                    "with tasks must yield at least one TestSpec",
                    severity="error",
                )
            )
        for spec in bundle.specs:
            violations.extend(TestSpecValidator.validate(spec, ir=ir).violations)

        report = PlanValidationReport.from_violations(
            target_kind="test_spec",
            target_id=target,
            violations=violations,
        )
        return self._persist_and_maybe_raise(
            ctx,
            report,
            f"test spec validation failed:\n{render_violations(report.violations)}",
            target=target,
        )

    @staticmethod
    def _load_ir(ctx: HarnessRunContext) -> PlanWorkflowIR | None:
        """Return the run's PlanWorkflowIR for cross-checking, or None (shallow)."""
        ir_ref = ctx.artifact_store.latest_by_kind("workflow_ir")
        if ir_ref is None:
            return None
        try:
            return PlanWorkflowIR.model_validate_json(ctx.artifact_store.get(ir_ref.id))
        except Exception:
            return None  # unparseable IR → fall back to shallow validation

    def _persist_and_maybe_raise(
        self,
        ctx: HarnessRunContext,
        report: PlanValidationReport,
        error_message: str,
        *,
        target: str,
    ) -> PlanArtifactRef:
        report_ref = ctx.artifact_store.put_json(
            kind="validation_report",
            obj=json.loads(report.model_dump_json()),
            created_by="ValidateTestSpec",
            parent_ids=[target],
        )
        if not report.passed and self._raise_on_failure:
            raise StagePersistedFailureError(report_ref, error_message)
        return report_ref
