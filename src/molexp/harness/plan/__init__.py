"""Emergent-plan task-board state: model, persistence, and freezing.

This subpackage owns the current-state-only :class:`TaskBoard` (a version
counter plus a tuple of :class:`BoardTask`), its pure transitions, its
full-rewrite persistence with an optimistic-version guard, and the
content-addressed freezing of an :class:`ExperimentPlan` (spec + board).

It is a subpackage **import target** only — it is deliberately NOT re-exported
from ``molexp.harness.__all__`` (the top-level surface stays locked). Reach
these symbols via ``molexp.harness.plan``.
"""

from __future__ import annotations

from molexp.harness.plan.board_store import (
    BoardVersionConflict,
    board_path,
    read_board,
    write_board,
)
from molexp.harness.plan.experiment_plan import (
    FROZEN_PLAN_KIND,
    FROZEN_SPEC_KIND,
    ExperimentPlan,
    freeze_experiment_plan,
    freeze_spec,
)
from molexp.harness.plan.task_board import (
    BoardTask,
    Difficulty,
    FeasibilityAnnotation,
    TaskBoard,
    TaskNotFoundError,
    TaskStatus,
    annotate_feasibility,
    place_task,
    remove_task,
    set_task_status,
)

__all__ = [
    "FROZEN_PLAN_KIND",
    "FROZEN_SPEC_KIND",
    "BoardTask",
    "BoardVersionConflict",
    "Difficulty",
    "ExperimentPlan",
    "FeasibilityAnnotation",
    "TaskBoard",
    "TaskNotFoundError",
    "TaskStatus",
    "annotate_feasibility",
    "board_path",
    "freeze_experiment_plan",
    "freeze_spec",
    "place_task",
    "read_board",
    "remove_task",
    "set_task_status",
    "write_board",
]
