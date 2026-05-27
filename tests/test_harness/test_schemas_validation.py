"""Tests for ValidationReport / ValidationViolation (Phase 3).

Locks the contract every validator returns:
- frozen pydantic round-trip
- target_kind is Literal["workflow_ir", "bound_workflow"]
- severity is Literal["error", "warning"], defaults to "error"
- ValidationReport.from_violations sets passed = not any(error severity)
"""

from __future__ import annotations

from typing import get_args, get_origin

import pytest
from pydantic import ValidationError


def test_validation_violation_round_trip() -> None:
    from molexp.harness.schemas.validation import ValidationViolation

    v = ValidationViolation(
        code="duplicate_task_id",
        message="task id 't1' appears twice",
        path="tasks[1].id",
        severity="error",
    )
    dumped = v.model_dump_json()
    rehydrated = ValidationViolation.model_validate_json(dumped)
    assert rehydrated == v


def test_validation_violation_defaults() -> None:
    from molexp.harness.schemas.validation import ValidationViolation

    v = ValidationViolation(code="x", message="y")
    assert v.path is None
    assert v.severity == "error"


def test_validation_violation_is_frozen() -> None:
    from molexp.harness.schemas.validation import ValidationViolation

    v = ValidationViolation(code="x", message="y")
    with pytest.raises(ValidationError):
        v.code = "mutated"  # type: ignore[misc]


def test_validation_violation_severity_rejects_unknown() -> None:
    from molexp.harness.schemas.validation import ValidationViolation

    with pytest.raises(ValidationError):
        ValidationViolation(code="x", message="y", severity="critical")  # type: ignore[arg-type]


def test_validation_violation_severity_is_literal_not_enum() -> None:
    from typing import Literal

    from molexp.harness.schemas.validation import ValidationViolation

    field = ValidationViolation.model_fields["severity"]
    assert get_origin(field.annotation) is Literal
    assert set(get_args(field.annotation)) == {"error", "warning"}


def test_validation_report_round_trip() -> None:
    from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

    report = ValidationReport(
        passed=False,
        violations=[ValidationViolation(code="x", message="y")],
        target_kind="workflow_ir",
        target_id="wf-001",
    )
    dumped = report.model_dump_json()
    rehydrated = ValidationReport.model_validate_json(dumped)
    assert rehydrated == report


def test_validation_report_target_kind_is_literal() -> None:
    from typing import Literal

    from molexp.harness.schemas.validation import ValidationReport

    field = ValidationReport.model_fields["target_kind"]
    assert get_origin(field.annotation) is Literal
    # Phase 5 widens additively. Old values stay; new values join.
    actual = set(get_args(field.annotation))
    assert {"workflow_ir", "bound_workflow"} <= actual
    assert {"test_spec", "provenance"} <= actual


def test_validation_report_target_kind_rejects_unknown() -> None:
    from molexp.harness.schemas.validation import ValidationReport

    with pytest.raises(ValidationError):
        ValidationReport(
            passed=True,
            violations=[],
            target_kind="not_a_kind",  # type: ignore[arg-type]
            target_id="x",
        )


def test_from_violations_empty_yields_passed() -> None:
    from molexp.harness.schemas.validation import ValidationReport

    report = ValidationReport.from_violations(
        target_kind="workflow_ir", target_id="wf-001", violations=[]
    )
    assert report.passed is True
    assert report.violations == []


def test_from_violations_warning_only_still_passes() -> None:
    from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

    report = ValidationReport.from_violations(
        target_kind="workflow_ir",
        target_id="wf-001",
        violations=[ValidationViolation(code="hint", message="m", severity="warning")],
    )
    assert report.passed is True
    assert len(report.violations) == 1


def test_from_violations_single_error_fails() -> None:
    from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

    report = ValidationReport.from_violations(
        target_kind="workflow_ir",
        target_id="wf-001",
        violations=[ValidationViolation(code="bad", message="m", severity="error")],
    )
    assert report.passed is False


def test_from_violations_mixed_warnings_and_error_fails() -> None:
    from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

    report = ValidationReport.from_violations(
        target_kind="bound_workflow",
        target_id="bw-001",
        violations=[
            ValidationViolation(code="hint", message="m", severity="warning"),
            ValidationViolation(code="bad", message="m", severity="error"),
        ],
    )
    assert report.passed is False


def test_validation_report_is_frozen() -> None:
    from molexp.harness.schemas.validation import ValidationReport

    report = ValidationReport(
        passed=True,
        violations=[],
        target_kind="workflow_ir",
        target_id="wf-001",
    )
    with pytest.raises(ValidationError):
        report.passed = False  # type: ignore[misc]
