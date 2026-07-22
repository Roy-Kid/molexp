"""Tests for ``TestSpecValidator`` (``validators/test_spec.py``).

One case per structural check (seven codes), a clean baseline, and the
shallow-vs-cross-checked two-mode contract (target codes fire only when an
``ir`` is supplied).
"""

from __future__ import annotations


def _baseline_test_spec():
    from molexp.harness.schemas.test_spec import TestSpec

    return TestSpec(
        id="ts-001",
        name="Schema sanity",
        kind="schema_test",
        description="Verify PlanWorkflowIR schema validates cleanly",
        target_workflow_id="wf-001",
    )


def _baseline_ir():
    from molexp.harness.schemas.workflow_ir import PlanTaskIR, PlanWorkflowIR

    return PlanWorkflowIR(
        id="wf-001",
        name="wf",
        objective="x",
        inputs={},
        tasks=[
            PlanTaskIR(
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


class TestTestSpecValidator:
    def test_baseline_test_spec_is_clean(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        report = TestSpecValidator.validate(_baseline_test_spec())
        assert report.passed is True
        assert report.violations == []
        assert report.target_kind == "test_spec"
        assert report.target_id == "ts-001"

    def test_missing_target_when_neither_task_nor_workflow(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(
            update={"target_task_id": None, "target_workflow_id": None}
        )
        report = TestSpecValidator.validate(spec)
        assert "missing_target" in _codes(report)
        assert report.passed is False

    def test_ambiguous_target_when_both_task_and_workflow(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(
            update={"target_task_id": "task_a", "target_workflow_id": "wf-001"}
        )
        report = TestSpecValidator.validate(spec)
        assert "ambiguous_target" in _codes(report)

    def test_unknown_task_target_fires_and_message_caps_candidates(self) -> None:
        """With an ir, an unknown task id fires ``unknown_task_target``; a huge
        candidate list is sorted, capped, and reports how many more (owns the
        ``format_candidates`` cap contract)."""
        from molexp.harness.schemas.workflow_ir import PlanTaskIR
        from molexp.harness.validators.test_spec import TestSpecValidator

        many_tasks = [
            PlanTaskIR(
                id=f"task_{i:02d}",
                name=f"Task {i}",
                purpose="x",
                task_type="x",
                inputs={},
                outputs={"out": "out.json"},
            )
            for i in range(30)
        ]
        ir = _baseline_ir().model_copy(update={"tasks": many_tasks})
        spec = _baseline_test_spec().model_copy(
            update={"target_task_id": "ghost_task", "target_workflow_id": None}
        )
        report = TestSpecValidator.validate(spec, ir=ir)
        [violation] = [v for v in report.violations if v.code == "unknown_task_target"]
        assert "task_00" in violation.message  # sorted list starts at the front
        assert "task_29" not in violation.message  # tail is capped away
        assert "15 more" in violation.message  # and the cap says how many more

    def test_unknown_task_target_stays_silent_without_ir(self) -> None:
        """Shallow mode (no ir) must NOT resolve target ids."""
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(
            update={"target_task_id": "ghost_task", "target_workflow_id": None}
        )
        report = TestSpecValidator.validate(spec)
        assert "unknown_task_target" not in _codes(report)

    def test_unknown_workflow_target_with_ir(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(update={"target_workflow_id": "wf-other"})
        report = TestSpecValidator.validate(spec, ir=_baseline_ir())
        assert "unknown_workflow_target" in _codes(report)

    def test_tolerance_requires_metric_is_a_warning(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(update={"tolerance": {"mobility": 0.05}})
        report = TestSpecValidator.validate(spec)
        matches = [v for v in report.violations if v.code == "tolerance_requires_metric"]
        assert matches, "expected tolerance_requires_metric warning"
        assert matches[0].severity == "warning"
        # Warning only → passed stays True.
        assert report.passed is True

    def test_command_with_shell_is_an_error(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(update={"command": ["bash", "-c", "echo hi"]})
        report = TestSpecValidator.validate(spec)
        assert "command_with_shell" in _codes(report)
        assert report.passed is False

    def test_safe_command_does_not_fire_shell_check(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(update={"command": ["pytest", "tests/foo.py"]})
        report = TestSpecValidator.validate(spec)
        assert "command_with_shell" not in _codes(report)

    def test_numerical_test_missing_tolerance_is_a_warning(self) -> None:
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(
            update={"kind": "numerical_tolerance_test", "tolerance": {}}
        )
        report = TestSpecValidator.validate(spec)
        matches = [v for v in report.violations if v.code == "numerical_test_missing_tolerance"]
        assert matches, "expected numerical_test_missing_tolerance warning"
        assert matches[0].severity == "warning"

    def test_numerical_test_with_matching_tolerance_is_clean(self) -> None:
        from molexp.harness.schemas.parameter import ParameterValue
        from molexp.harness.validators.test_spec import TestSpecValidator

        spec = _baseline_test_spec().model_copy(
            update={
                "kind": "numerical_tolerance_test",
                "expected_metrics": {"x": ParameterValue(value=1.0, source="user_provided")},
                "tolerance": {"x": 0.05},
            }
        )
        report = TestSpecValidator.validate(spec)
        assert "numerical_test_missing_tolerance" not in _codes(report)
        assert "tolerance_requires_metric" not in _codes(report)
