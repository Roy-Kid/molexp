"""Tests for PlanValidationReport / ValidationViolation (Phase 3).

Locks the contract every validator returns:
- PlanValidationReport.from_violations sets passed = not any(error severity)
- target_kind Literal widens additively (old values stay when new ones join)
"""

from __future__ import annotations

from typing import get_args, get_origin


def test_validation_report_target_kind_is_literal() -> None:
    from typing import Literal

    from molexp.harness.schemas.validation import PlanValidationReport

    field = PlanValidationReport.model_fields["target_kind"]
    assert get_origin(field.annotation) is Literal
    # Phase 5 widens additively. Old values stay; new values join.
    actual = set(get_args(field.annotation))
    assert {"workflow_ir", "bound_workflow"} <= actual
    assert {"test_spec", "provenance"} <= actual


def test_from_violations_empty_yields_passed() -> None:
    from molexp.harness.schemas.validation import PlanValidationReport

    report = PlanValidationReport.from_violations(
        target_kind="workflow_ir", target_id="wf-001", violations=[]
    )
    assert report.passed is True
    assert report.violations == []


def test_from_violations_warning_only_still_passes() -> None:
    from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation

    report = PlanValidationReport.from_violations(
        target_kind="workflow_ir",
        target_id="wf-001",
        violations=[ValidationViolation(code="hint", message="m", severity="warning")],
    )
    assert report.passed is True
    assert len(report.violations) == 1


def test_from_violations_mixed_warnings_and_error_fails() -> None:
    from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation

    report = PlanValidationReport.from_violations(
        target_kind="bound_workflow",
        target_id="bw-001",
        violations=[
            ValidationViolation(code="hint", message="m", severity="warning"),
            ValidationViolation(code="bad", message="m", severity="error"),
        ],
    )
    assert report.passed is False
