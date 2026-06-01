"""Tests for TestSpec / TestResult / TestKind / TestStatus (Phase 5 §4.8).

Locks the wire format:
- frozen pydantic round-trip
- TestKind Literal carries all 9 values from harness-goal.md §4.8
- TestStatus Literal carries the 4 status values
- TestSpec / TestResult defaults
- ValidationReport.target_kind Literal widens additively
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args, get_origin

import pytest
from pydantic import ValidationError


def _minimal_test_spec_kwargs() -> dict:
    return {
        "id": "ts-001",
        "name": "Schema sanity",
        "kind": "schema_test",
        "description": "Verify WorkflowIR schema validates cleanly",
        "target_workflow_id": "wf-001",
    }


# ------------------------------------------------------------ TestKind / Status


def test_test_kind_carries_all_nine_values() -> None:
    from typing import Literal

    from molexp.harness.schemas.test_spec import TestKind

    assert get_origin(TestKind) is Literal
    expected = {
        "schema_test",
        "unit_test",
        "dry_run_test",
        "integration_test",
        "regression_test",
        "numerical_tolerance_test",
        "artifact_existence_test",
        "provenance_test",
        "resource_policy_test",
    }
    assert set(get_args(TestKind)) == expected


def test_test_status_carries_all_four_values() -> None:
    from molexp.harness.schemas.test_spec import TestStatus

    assert set(get_args(TestStatus)) == {"passed", "failed", "skipped", "error"}


# ------------------------------------------------------------- TestSpec


def test_test_spec_minimal_round_trip() -> None:
    from molexp.harness.schemas.test_spec import TestSpec

    spec = TestSpec(**_minimal_test_spec_kwargs())
    dumped = spec.model_dump_json()
    rehydrated = TestSpec.model_validate_json(dumped)
    assert rehydrated == spec


def test_test_spec_full_round_trip() -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.test_spec import TestSpec

    spec = TestSpec(
        id="ts-002",
        name="Mobility tolerance",
        kind="numerical_tolerance_test",
        target_task_id="analyze_trajectory",
        description="Mobility within 5% of reference",
        inputs={"trajectory": ParameterValue(value="traj.dcd", source="user_provided")},
        command=None,
        expected_artifacts=["mobility.json"],
        expected_metrics={
            "mobility": ParameterValue(
                value=1.2e-4, source="literature_default", citation="doi:foo"
            ),
        },
        tolerance={"mobility": 0.05},
        required=True,
    )
    dumped = spec.model_dump_json()
    rehydrated = TestSpec.model_validate_json(dumped)
    assert rehydrated == spec


def test_test_spec_defaults() -> None:
    from molexp.harness.schemas.test_spec import TestSpec

    spec = TestSpec(**_minimal_test_spec_kwargs())
    assert spec.target_task_id is None
    assert spec.inputs == {}
    assert spec.command is None
    assert spec.expected_artifacts == []
    assert spec.expected_metrics == {}
    assert spec.tolerance == {}
    assert spec.required is True


def test_test_spec_is_frozen() -> None:
    from molexp.harness.schemas.test_spec import TestSpec

    spec = TestSpec(**_minimal_test_spec_kwargs())
    with pytest.raises(ValidationError):
        spec.name = "mutated"  # type: ignore[misc]


def test_test_spec_rejects_unknown_kind() -> None:
    from molexp.harness.schemas.test_spec import TestSpec

    with pytest.raises(ValidationError):
        TestSpec(
            id="x",
            name="x",
            kind="not_a_real_kind",  # type: ignore[arg-type]
            description="x",
            target_workflow_id="x",
        )


def test_test_spec_default_factories_are_independent() -> None:
    from molexp.harness.schemas.test_spec import TestSpec

    a = TestSpec(**_minimal_test_spec_kwargs())
    b = TestSpec(**_minimal_test_spec_kwargs())
    assert a.inputs is not b.inputs
    assert a.expected_artifacts is not b.expected_artifacts
    assert a.tolerance is not b.tolerance


# ----------------------------------------------------------- TestResult


def _ref():
    from molexp.harness.schemas.artifact import ArtifactRef

    return ArtifactRef(
        id="art01234",
        kind="log",
        uri="file:///tmp/log",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )


def test_test_result_minimal_round_trip() -> None:
    from molexp.harness.schemas.test_spec import TestResult

    result = TestResult(id="tr-001", test_spec_id="ts-001", status="passed")
    dumped = result.model_dump_json()
    rehydrated = TestResult.model_validate_json(dumped)
    assert rehydrated == result


def test_test_result_full_round_trip() -> None:
    from molexp.harness.schemas.test_spec import TestResult

    result = TestResult(
        id="tr-002",
        test_spec_id="ts-002",
        status="failed",
        metrics={"mobility": 1.5e-4},
        produced_artifacts=[_ref()],
        stdout=_ref(),
        stderr=_ref(),
        reason="mobility outside tolerance",
    )
    dumped = result.model_dump_json()
    rehydrated = TestResult.model_validate_json(dumped)
    assert rehydrated == result


def test_test_result_defaults() -> None:
    from molexp.harness.schemas.test_spec import TestResult

    result = TestResult(id="tr-x", test_spec_id="ts-x", status="passed")
    assert result.metrics == {}
    assert result.produced_artifacts == []
    assert result.stdout is None
    assert result.stderr is None
    assert result.reason is None


def test_test_result_is_frozen() -> None:
    from molexp.harness.schemas.test_spec import TestResult

    result = TestResult(id="tr-x", test_spec_id="ts-x", status="passed")
    with pytest.raises(ValidationError):
        result.status = "failed"  # type: ignore[misc]


def test_test_result_rejects_unknown_status() -> None:
    from molexp.harness.schemas.test_spec import TestResult

    with pytest.raises(ValidationError):
        TestResult(
            id="tr-x",
            test_spec_id="ts-x",
            status="unknown",  # type: ignore[arg-type]
        )


# -------------------------------------------- ValidationReport widening


def test_validation_report_target_kind_widens_additively() -> None:
    from molexp.harness.schemas.validation import ValidationReport

    # New values accepted.
    ValidationReport(target_kind="test_spec", target_id="x", passed=True)
    ValidationReport(target_kind="provenance", target_id="x", passed=True)
    # Old values still accepted (regression guard).
    ValidationReport(target_kind="workflow_ir", target_id="x", passed=True)
    ValidationReport(target_kind="bound_workflow", target_id="x", passed=True)
    # Unknown still rejected.
    with pytest.raises(ValidationError):
        ValidationReport(target_kind="banana", target_id="x", passed=True)  # type: ignore[arg-type]


# ---------------------------------------------------------- re-exports


def test_phase5_schemas_re_exported_from_schemas_package() -> None:
    from molexp.harness.schemas import (
        TestKind as via_pkg_kind,
    )
    from molexp.harness.schemas import (
        TestResult as via_pkg_result,
    )
    from molexp.harness.schemas import (
        TestSpec as via_pkg_spec,
    )
    from molexp.harness.schemas import (
        TestStatus as via_pkg_status,
    )
    from molexp.harness.schemas.test_spec import (
        TestKind as via_mod_kind,
    )
    from molexp.harness.schemas.test_spec import (
        TestResult as via_mod_result,
    )
    from molexp.harness.schemas.test_spec import (
        TestSpec as via_mod_spec,
    )
    from molexp.harness.schemas.test_spec import (
        TestStatus as via_mod_status,
    )

    assert via_pkg_kind is via_mod_kind
    assert via_pkg_status is via_mod_status
    assert via_pkg_spec is via_mod_spec
    assert via_pkg_result is via_mod_result


def test_phase5_schemas_re_exported_from_top_level() -> None:
    from molexp.harness import TestKind, TestResult, TestSpec, TestStatus  # noqa: F401
