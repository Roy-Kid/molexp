"""Tests for validate_test_spec (Phase 5).

Seven codes + clean baseline + shallow vs cross-checked mode.

Codes:
- missing_target (error)
- ambiguous_target (error)
- unknown_task_target (error) — only fires when ir is provided
- unknown_workflow_target (error) — only fires when ir is provided
- tolerance_requires_metric (warning)
- command_with_shell (error)
- numerical_test_missing_tolerance (warning)
"""

from __future__ import annotations


def _baseline_test_spec():
    from molexp.harness.schemas.test_spec import TestSpec

    return TestSpec(
        id="ts-001",
        name="Schema sanity",
        kind="schema_test",
        description="Verify WorkflowIR schema validates cleanly",
        target_workflow_id="wf-001",
    )


def _baseline_ir():
    from molexp.harness.schemas.workflow_ir import TaskIR, WorkflowIR

    return WorkflowIR(
        id="wf-001",
        name="wf",
        objective="x",
        inputs={},
        tasks=[
            TaskIR(
                id="task_a",
                name="Task A",
                purpose="x",
                task_type="x",
                inputs={},
                outputs={"out": "out.json"},
            ),
        ],
        edges=[],
        expected_outputs=[],
    )


def _codes(report) -> list[str]:
    return [v.code for v in report.violations]


# -------------------------------------------------------------- baseline


def test_baseline_test_spec_is_clean() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    report = validate_test_spec(_baseline_test_spec())
    assert report.passed is True
    assert report.violations == []
    assert report.target_kind == "test_spec"
    assert report.target_id == "ts-001"


def test_validate_test_spec_signature_and_import() -> None:
    """Importable from both paths; returns ValidationReport."""
    from molexp.harness import validate_test_spec as top
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.validators import validate_test_spec as via_pkg
    from molexp.harness.validators.test_spec import validate_test_spec as via_mod

    assert top is via_pkg is via_mod
    report = top(_baseline_test_spec())
    assert isinstance(report, ValidationReport)


# -------------------------------------------------------------- codes


def test_missing_target() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={"target_task_id": None, "target_workflow_id": None}
    )
    report = validate_test_spec(spec)
    assert "missing_target" in _codes(report)
    assert report.passed is False


def test_ambiguous_target() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={"target_task_id": "task_a", "target_workflow_id": "wf-001"}
    )
    report = validate_test_spec(spec)
    assert "ambiguous_target" in _codes(report)


def test_unknown_task_target_with_ir() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={"target_task_id": "ghost_task", "target_workflow_id": None}
    )
    report = validate_test_spec(spec, ir=_baseline_ir())
    assert "unknown_task_target" in _codes(report)


def test_shallow_mode_skips_unknown_task_target() -> None:
    """Without ir, unknown_task_target must NOT fire."""
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={"target_task_id": "ghost_task", "target_workflow_id": None}
    )
    report = validate_test_spec(spec)
    assert "unknown_task_target" not in _codes(report)


def test_cross_check_mode_clean_when_target_resolves() -> None:
    """target_task_id present in ir → no violation."""
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={"target_task_id": "task_a", "target_workflow_id": None}
    )
    report = validate_test_spec(spec, ir=_baseline_ir())
    assert "unknown_task_target" not in _codes(report)


def test_unknown_workflow_target_with_ir() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(update={"target_workflow_id": "wf-other"})
    report = validate_test_spec(spec, ir=_baseline_ir())
    assert "unknown_workflow_target" in _codes(report)


def test_shallow_mode_skips_unknown_workflow_target() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(update={"target_workflow_id": "wf-other"})
    report = validate_test_spec(spec)
    assert "unknown_workflow_target" not in _codes(report)


def test_tolerance_requires_metric_warning() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(update={"tolerance": {"mobility": 0.05}})
    report = validate_test_spec(spec)
    matches = [v for v in report.violations if v.code == "tolerance_requires_metric"]
    assert matches, "expected tolerance_requires_metric warning"
    assert matches[0].severity == "warning"
    # Warning only → passed=True if no error.
    if all(v.severity == "warning" for v in report.violations):
        assert report.passed is True


def test_command_with_shell_error() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(update={"command": ["bash", "-c", "echo hi"]})
    report = validate_test_spec(spec)
    assert "command_with_shell" in _codes(report)
    assert report.passed is False


def test_command_with_subprocess_run_error() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={"command": ["python", "-c", "subprocess.run(['rm', '-rf', '/'])"]}
    )
    report = validate_test_spec(spec)
    assert "command_with_shell" in _codes(report)


def test_command_with_semicolon_error() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(update={"command": ["mytool", "a;b"]})
    report = validate_test_spec(spec)
    assert "command_with_shell" in _codes(report)


def test_clean_command_no_violation() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(update={"command": ["pytest", "tests/foo.py"]})
    report = validate_test_spec(spec)
    assert "command_with_shell" not in _codes(report)


def test_numerical_test_missing_tolerance_warning() -> None:
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={"kind": "numerical_tolerance_test", "tolerance": {}}
    )
    report = validate_test_spec(spec)
    matches = [v for v in report.violations if v.code == "numerical_test_missing_tolerance"]
    assert matches, "expected numerical_test_missing_tolerance warning"
    assert matches[0].severity == "warning"


def test_numerical_test_with_tolerance_clean() -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.validators.test_spec import validate_test_spec

    spec = _baseline_test_spec().model_copy(
        update={
            "kind": "numerical_tolerance_test",
            "expected_metrics": {"x": ParameterValue(value=1.0, source="user_provided")},
            "tolerance": {"x": 0.05},
        }
    )
    report = validate_test_spec(spec)
    assert "numerical_test_missing_tolerance" not in _codes(report)
    assert "tolerance_requires_metric" not in _codes(report)
