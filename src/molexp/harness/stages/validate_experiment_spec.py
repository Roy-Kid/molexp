"""``ValidateExperimentSpec`` — validate the concrete spec (plan step 2).

Loads the latest ``experiment_spec`` artifact (and the ``experiment_report``
it derives from, for coverage checks), runs the pure
:class:`ExperimentSpecValidator`, and persists a ``validation_report``
artifact unconditionally — failed validations stay first-class in the audit
trail. Only after persistence does it lift to :class:`StagePersistedFailureError`
when ``raise_on_failure`` (default True) and the report failed. A
JSON/schema parse error on the spec is itself recorded as a one-violation
report before raising.
"""

from __future__ import annotations

import json
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StagePersistedFailureError
from molexp.harness.schemas import (
    ArtifactRef,
    ExperimentReport,
    ExperimentSpec,
    ValidationReport,
    ValidationViolation,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.validators.experiment_spec import ExperimentSpecValidator

__all__ = ["ValidateExperimentSpec"]


class ValidateExperimentSpec(Stage):
    """Validate an ExperimentSpec artifact; persist a ValidationReport artifact."""

    name: ClassVar[str] = "validate_experiment_spec"

    def __init__(self, *, raise_on_failure: bool = True) -> None:
        self._raise_on_failure = raise_on_failure

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        spec_ref = require_latest(ctx, "experiment_spec", stage=self.name)
        target_id = spec_ref.id

        try:
            spec = ExperimentSpec.model_validate_json(ctx.artifact_store.get(target_id))
        except Exception as exc:
            report = ValidationReport.from_violations(
                target_kind="experiment_spec",
                target_id=target_id,
                violations=[
                    ValidationViolation(
                        code="experiment_spec_parse_error",
                        message=f"ExperimentSpec JSON failed schema validation: {exc!r}",
                        severity="error",
                    )
                ],
            )
            report_ref = ctx.artifact_store.put_json(
                kind="validation_report",
                obj=json.loads(report.model_dump_json()),
                created_by="ValidateExperimentSpec",
                parent_ids=[target_id],
            )
            if self._raise_on_failure:
                raise StagePersistedFailureError(
                    report_ref, f"ExperimentSpec parse failed: {exc!r}"
                ) from exc
            return report_ref

        report_in = ctx.artifact_store.latest_by_kind("experiment_report")
        report = None
        if report_in is not None:
            try:
                report = ExperimentReport.model_validate_json(ctx.artifact_store.get(report_in.id))
            except Exception:
                report = None

        result = ExperimentSpecValidator.validate(spec, report=report)
        report_ref = ctx.artifact_store.put_json(
            kind="validation_report",
            obj=json.loads(result.model_dump_json()),
            created_by="ValidateExperimentSpec",
            parent_ids=[target_id],
        )
        if not result.passed and self._raise_on_failure:
            error_codes = [v.code for v in result.violations if v.severity == "error"]
            raise StagePersistedFailureError(
                report_ref, f"ExperimentSpec validation failed with violations: {error_codes}"
            )
        return report_ref
