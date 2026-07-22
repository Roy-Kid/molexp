"""Tests for ``PlanValidationReport`` — the validator return contract.

Every harness validator returns a :class:`PlanValidationReport` whose ``passed``
flag is derived: ``True`` iff zero ``severity="error"`` violations exist —
warning-only reports still pass.
"""

from __future__ import annotations

from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation


class TestPlanValidationReport:
    def test_from_violations_passes_when_only_warnings(self) -> None:
        report = PlanValidationReport.from_violations(
            target_kind="workflow_ir",
            target_id="wf-001",
            violations=[ValidationViolation(code="hint", message="m", severity="warning")],
        )
        assert report.passed is True
        assert len(report.violations) == 1

    def test_from_violations_fails_when_any_error(self) -> None:
        report = PlanValidationReport.from_violations(
            target_kind="bound_workflow",
            target_id="bw-001",
            violations=[
                ValidationViolation(code="hint", message="m", severity="warning"),
                ValidationViolation(code="bad", message="m", severity="error"),
            ],
        )
        assert report.passed is False
