"""``ValidateInputSet`` — validate the parameter-space spec (plan step 6).

Loads the latest ``input_set`` artifact (and the ``workflow_ir`` it expands,
for axis-name checks), runs the pure :class:`InputSetValidator`, and persists
a ``validation_report`` artifact unconditionally. Lifts to
:class:`StagePersistedFailureError` when ``raise_on_failure`` (default True)
and the report failed. A JSON/schema parse error on the input set is itself
recorded as a one-violation report before raising.
"""

from __future__ import annotations

import json
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StagePersistedFailureError
from molexp.harness.schemas import (
    ArtifactRef,
    InputSet,
    ValidationReport,
    ValidationViolation,
    WorkflowIR,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.validators.input_set import InputSetValidator

__all__ = ["ValidateInputSet"]


class ValidateInputSet(Stage):
    """Validate an InputSet artifact; persist a ValidationReport artifact."""

    name: ClassVar[str] = "validate_input_set"

    def __init__(self, *, raise_on_failure: bool = True) -> None:
        self._raise_on_failure = raise_on_failure

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        input_set_ref = require_latest(ctx, "input_set", stage=self.name)
        target_id = input_set_ref.id

        try:
            input_set = InputSet.model_validate_json(ctx.artifact_store.get(target_id))
        except Exception as exc:
            report = ValidationReport.from_violations(
                target_kind="input_set",
                target_id=target_id,
                violations=[
                    ValidationViolation(
                        code="input_set_parse_error",
                        message=f"InputSet JSON failed schema validation: {exc!r}",
                        severity="error",
                    )
                ],
            )
            report_ref = ctx.artifact_store.put_json(
                kind="validation_report",
                obj=json.loads(report.model_dump_json()),
                created_by="ValidateInputSet",
                parent_ids=[target_id],
            )
            if self._raise_on_failure:
                raise StagePersistedFailureError(
                    report_ref, f"InputSet parse failed: {exc!r}"
                ) from exc
            return report_ref

        ir_in = ctx.artifact_store.latest_by_kind("workflow_ir")
        ir = None
        if ir_in is not None:
            try:
                ir = WorkflowIR.model_validate_json(ctx.artifact_store.get(ir_in.id))
            except Exception:
                ir = None

        result = InputSetValidator.validate(input_set, ir=ir)
        report_ref = ctx.artifact_store.put_json(
            kind="validation_report",
            obj=json.loads(result.model_dump_json()),
            created_by="ValidateInputSet",
            parent_ids=[target_id],
        )
        if not result.passed and self._raise_on_failure:
            error_codes = [v.code for v in result.violations if v.severity == "error"]
            raise StagePersistedFailureError(
                report_ref, f"InputSet validation failed with violations: {error_codes}"
            )
        return report_ref
