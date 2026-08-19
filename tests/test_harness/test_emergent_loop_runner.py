"""Form-check is a workflow edge, not an outer agent loop."""

from __future__ import annotations

from molexp.harness.plan import (
    BoardTask,
    Difficulty,
    ExperimentPlan,
    FeasibilityAnnotation,
    TaskBoard,
)
from molexp.harness.validators import PlanFormValidator

_SPEC = {"title": "Zwitterion CG", "objective": "measure area-per-lipid"}


def _valid_board() -> TaskBoard:
    return TaskBoard(
        version=1,
        tasks=(
            BoardTask(
                id="t1",
                name="build system",
                acceptance=("energy is finite",),
                feasibility=FeasibilityAnnotation(reachable=True, difficulty=Difficulty.TRIVIAL),
            ),
        ),
    )


def _malformed_board() -> TaskBoard:
    return TaskBoard(version=1, tasks=(BoardTask(id="t1", name="build", acceptance=()),))


class TestFormValidator:
    def test_invalid_board_fails_and_valid_board_passes(self) -> None:
        bad = PlanFormValidator.validate(ExperimentPlan(spec=_SPEC, board=_malformed_board()))
        assert not bad.passed
        good = PlanFormValidator.validate(ExperimentPlan(spec=_SPEC, board=_valid_board()))
        assert good.passed
