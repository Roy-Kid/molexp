"""RED unit tests for :class:`EmergentPlanFormValidator` (spec plan-emergent-05b-guard).

Mirrors the pure-function / one-code-per-defect / never-raises pattern of
:class:`molexp.harness.validators.workflow_ir.WorkflowIRValidator`. The validator
grades an :class:`~molexp.harness.plan.ExperimentPlan` (spec + task board) and
returns a :class:`~molexp.harness.schemas.validation.PlanValidationReport` with
``target_kind="experiment_plan"`` (a new ``_TARGET_KINDS`` member).

These tests are written before the production symbol exists, so importing
``EmergentPlanFormValidator`` fails RED (ImportError) until it is authored.
"""

from __future__ import annotations

from typing import Any

from molexp.harness.plan import (
    BoardTask,
    Difficulty,
    ExperimentPlan,
    FeasibilityAnnotation,
    TaskBoard,
    TaskStatus,  # noqa: F401  (documented lifecycle vocab)
)
from molexp.harness.validators.emergent_plan_form import EmergentPlanFormValidator

# --------------------------------------------------------------- builders


def _reachable() -> FeasibilityAnnotation:
    return FeasibilityAnnotation(
        reachable=True,
        difficulty=Difficulty.TRIVIAL,
        rationale="grounded",
        probed_refs=("molpy.build.run",),
    )


def _task(
    task_id: str = "t1",
    *,
    name: str = "Build system",
    acceptance: tuple[str, ...] = ("beads placed",),
    feasibility: FeasibilityAnnotation | None = None,
) -> BoardTask:
    return BoardTask(
        id=task_id,
        name=name,
        acceptance=acceptance,
        feasibility=feasibility if feasibility is not None else _reachable(),
    )


def _plan(
    *,
    spec: dict[str, Any] | None = None,
    tasks: tuple[BoardTask, ...] | None = None,
) -> ExperimentPlan:
    return ExperimentPlan(
        spec={"id": "exp-1", "title": "T", "objective": "O"} if spec is None else spec,
        board=TaskBoard(tasks=(_task(),) if tasks is None else tasks),
    )


def _codes(report: object) -> list[str]:
    return [v.code for v in report.violations]  # type: ignore[attr-defined]


def _error_codes(report: object) -> list[str]:
    return [v.code for v in report.violations if v.severity == "error"]  # type: ignore[attr-defined]


class TestEmergentPlanFormValidatorWellFormed:
    def test_well_formed_plan_passes_clean(self) -> None:
        report = EmergentPlanFormValidator.validate(_plan())
        assert report.passed is True
        assert report.violations == []

    def test_target_kind_is_experiment_plan(self) -> None:
        report = EmergentPlanFormValidator.validate(_plan())
        assert report.target_kind == "experiment_plan"

    def test_target_id_prefers_spec_id(self) -> None:
        report = EmergentPlanFormValidator.validate(_plan())
        assert report.target_id == "exp-1"

    def test_target_id_falls_back_to_title_then_default(self) -> None:
        titled = EmergentPlanFormValidator.validate(_plan(spec={"title": "Zwitterion"}))
        assert titled.target_id == "Zwitterion"
        default = EmergentPlanFormValidator.validate(
            ExperimentPlan(spec={}, board=TaskBoard(tasks=(_task(),)))
        )
        assert default.target_id == "experiment_plan"


class TestSpecIncomplete:
    def test_empty_spec_is_incomplete(self) -> None:
        report = EmergentPlanFormValidator.validate(
            ExperimentPlan(spec={}, board=TaskBoard(tasks=(_task(),)))
        )
        assert "spec_incomplete" in _codes(report)
        assert report.passed is False

    def test_missing_title_and_objective_report_one_each(self) -> None:
        report = EmergentPlanFormValidator.validate(_plan(spec={"id": "exp-1"}))
        assert _codes(report).count("spec_incomplete") == 2
        assert report.passed is False

    def test_blank_objective_is_incomplete(self) -> None:
        report = EmergentPlanFormValidator.validate(
            _plan(spec={"id": "exp-1", "title": "T", "objective": ""})
        )
        assert "spec_incomplete" in _codes(report)
        assert report.passed is False


class TestBoardDefects:
    def test_empty_board_blocks(self) -> None:
        report = EmergentPlanFormValidator.validate(_plan(tasks=()))
        assert "empty_board" in _codes(report)
        assert report.passed is False

    def test_duplicate_task_id_blocks(self) -> None:
        report = EmergentPlanFormValidator.validate(
            _plan(tasks=(_task("dup"), _task("dup", name="Second")))
        )
        assert "duplicate_task_id" in _codes(report)
        assert report.passed is False

    def test_blank_task_id_blocks(self) -> None:
        report = EmergentPlanFormValidator.validate(_plan(tasks=(_task(""),)))
        assert "blank_task_id" in _codes(report)
        assert report.passed is False

    def test_task_missing_acceptance_blocks(self) -> None:
        report = EmergentPlanFormValidator.validate(_plan(tasks=(_task(acceptance=()),)))
        assert "task_missing_acceptance" in _codes(report)
        assert report.passed is False

    def test_task_missing_feasibility_blocks(self) -> None:
        report = EmergentPlanFormValidator.validate(
            _plan(tasks=(BoardTask(id="t1", name="Build", acceptance=("crit",)),))
        )
        assert "task_missing_feasibility" in _codes(report)
        assert report.passed is False


class TestUnreachableWarning:
    def test_unreachable_task_warns_but_still_passes(self) -> None:
        unreachable = FeasibilityAnnotation(
            reachable=False, difficulty=Difficulty.HARD, rationale="no capability"
        )
        report = EmergentPlanFormValidator.validate(_plan(tasks=(_task(feasibility=unreachable),)))
        assert "task_unreachable" in _codes(report)
        assert _error_codes(report) == []
        assert report.passed is True


class TestNeverRaises:
    def test_degenerate_plan_does_not_raise(self) -> None:
        report = EmergentPlanFormValidator.validate(
            ExperimentPlan(spec={}, board=TaskBoard(tasks=()))
        )
        assert report.passed is False
        assert "empty_board" in _codes(report)
        assert "spec_incomplete" in _codes(report)


class TestExportedFromValidatorsPackage:
    def test_symbol_is_re_exported(self) -> None:
        from molexp.harness.validators import EmergentPlanFormValidator as Exported

        assert Exported is EmergentPlanFormValidator
