"""RED tests for ``molexp.harness.plan.experiment_plan`` — freeze determinism.

Pins content-addressed freezing of an ``ExperimentPlan`` (spec + board) and a
raw spec dict through a real ``FileArtifactStore``: the same content freezes to
the same ``PlanArtifactRef.id`` + ``sha256`` (idempotent + content-addressed),
a board change flips the id, and ``ExperimentPlan`` forbids extra fields.

Function-level unit tests only (real ``FileArtifactStore(tmp_path)``, no
subprocess). The production subpackage ``molexp.harness.plan`` does not exist
yet: a ``ModuleNotFoundError`` at import time is the valid RED signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.harness.plan import (
    FROZEN_PLAN_KIND,
    FROZEN_SPEC_KIND,
    BoardTask,
    ExperimentPlan,
    TaskBoard,
    freeze_experiment_plan,
    freeze_spec,
)
from molexp.harness.store.file_artifact_store import FileArtifactStore


def _plan() -> ExperimentPlan:
    """A fixed, content-stable plan (identical field/key ordering per call)."""
    return ExperimentPlan(
        spec={"title": "demo", "steps": 3},
        board=TaskBoard(version=1, tasks=(BoardTask(id="t1", name="build"),)),
    )


class TestKindConstants:
    def test_kind_constant_values(self) -> None:
        assert FROZEN_SPEC_KIND == "frozen_experiment_spec"
        assert FROZEN_PLAN_KIND == "frozen_experiment_plan"


class TestFreezeExperimentPlan:
    def test_freezing_same_plan_twice_is_idempotent(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path)
        plan = _plan()
        first = freeze_experiment_plan(plan, store, created_by="planner")
        second = freeze_experiment_plan(plan, store, created_by="planner")
        assert first.id == second.id
        assert first.sha256 == second.sha256

    def test_content_equal_plans_freeze_to_same_id(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path)
        first = freeze_experiment_plan(_plan(), store, created_by="planner")
        second = freeze_experiment_plan(_plan(), store, created_by="planner")
        assert first.id == second.id

    def test_board_change_changes_id(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path)
        base = freeze_experiment_plan(_plan(), store, created_by="planner")
        changed = ExperimentPlan(
            spec={"title": "demo", "steps": 3},
            board=TaskBoard(
                version=2,
                tasks=(BoardTask(id="t1", name="build"), BoardTask(id="t2", name="test")),
            ),
        )
        ref = freeze_experiment_plan(changed, store, created_by="planner")
        assert ref.id != base.id

    def test_frozen_plan_uses_frozen_plan_kind(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path)
        ref = freeze_experiment_plan(_plan(), store, created_by="planner")
        assert ref.kind == FROZEN_PLAN_KIND


class TestFreezeSpec:
    def test_freezing_same_spec_twice_is_idempotent(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path)
        spec = {"title": "demo", "n": 5}
        first = freeze_spec(spec, store, created_by="planner")
        second = freeze_spec(spec, store, created_by="planner")
        assert first.id == second.id
        assert first.sha256 == second.sha256

    def test_different_spec_changes_id(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path)
        first = freeze_spec({"title": "demo"}, store, created_by="planner")
        second = freeze_spec({"title": "other"}, store, created_by="planner")
        assert first.id != second.id

    def test_frozen_spec_uses_frozen_spec_kind(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path)
        ref = freeze_spec({"x": 1}, store, created_by="planner")
        assert ref.kind == FROZEN_SPEC_KIND


class TestExperimentPlanExtraForbid:
    def test_valid_construction(self) -> None:
        plan = ExperimentPlan(spec={"a": 1}, board=TaskBoard())
        assert plan.spec == {"a": 1}
        assert plan.board == TaskBoard()

    def test_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentPlan(spec={}, board=TaskBoard(), workflow_source="x")  # type: ignore[call-arg]
