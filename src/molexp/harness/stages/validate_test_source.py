"""``ValidateTestSource`` — static gate on generated pytest source.

Loads the latest ``test_source`` artifact and runs the pure
:func:`validate_test_source` pre-checks (syntax, public-surface imports,
test-function presence, byte-compile). Unlike
:class:`ValidateWorkflowSource` there is **no** compile-through-the-engine
step and **no** ``exec`` — actually running the tests is
:class:`ExecuteTests`'s job, through a harness executor subprocess. A
:class:`ValidationReport` is always persisted; on failure the stage raises
:class:`StagePersistedFailureError` after persisting.
"""

from __future__ import annotations

import json
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StagePersistedFailureError
from molexp.harness.schemas import (
    ArtifactRef,
    TestSource,
    ValidationReport,
    ValidationViolation,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.validators.test_source import validate_test_source

__all__ = ["ValidateTestSource"]


class ValidateTestSource(Stage):
    """Statically validate a TestSource artifact; persist a report."""

    name: ClassVar[str] = "validate_test_source"

    def __init__(self, *, raise_on_failure: bool = True) -> None:
        self._raise_on_failure = raise_on_failure

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        target = require_latest(ctx, "test_source", stage=self.name).id
        raw = ctx.artifact_store.get(target)

        try:
            ts = TestSource.model_validate_json(raw)
        except Exception as exc:
            report = ValidationReport.from_violations(
                target_kind="test_source",
                target_id=target,
                violations=[
                    ValidationViolation(
                        code="test_source_parse_error",
                        message=f"TestSource JSON failed schema validation: {exc!r}",
                        severity="error",
                    )
                ],
            )
            return self._persist_and_maybe_raise(
                ctx, report, f"TestSource parse failed: {exc!r}", target=target
            )

        report = validate_test_source(ts.source, target_id=target)
        codes = [v.code for v in report.violations if v.severity == "error"]
        return self._persist_and_maybe_raise(
            ctx, report, f"test source validation failed: {codes}", target=target
        )

    def _persist_and_maybe_raise(
        self,
        ctx: HarnessRunContext,
        report: ValidationReport,
        error_message: str,
        *,
        target: str,
    ) -> ArtifactRef:
        report_ref = ctx.artifact_store.put_json(
            kind="validation_report",
            obj=json.loads(report.model_dump_json()),
            created_by="ValidateTestSource",
            parent_ids=[target],
        )
        if not report.passed and self._raise_on_failure:
            raise StagePersistedFailureError(report_ref, error_message)
        return report_ref
