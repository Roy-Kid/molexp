"""RED unit tests for :class:`PlanReachabilityProbe` (spec plan-emergent-05b-guard).

The probe is a SYNC, read-only, artifact-free annotator: given a
:class:`~molexp.harness.plan.TaskBoard` and an optional
:class:`~molexp.harness.registry.CapabilityRegistry`, it returns a NEW board
whose tasks carry a :class:`~molexp.harness.plan.FeasibilityAnnotation` derived
from ``registry.search(task.name)`` via a deterministic difficulty ladder. It
NEVER mutates its input, takes NO store, and produces ZERO artifacts.

Written before the production symbol exists — importing
``PlanReachabilityProbe`` fails RED (ImportError) until it is authored.
"""

from __future__ import annotations

import inspect

from molexp.harness.plan import BoardTask, Difficulty, TaskBoard, TaskStatus
from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry
from molexp.harness.schemas import ToolCapability
from molexp.harness.stages.plan_reachability_probe import PlanReachabilityProbe


# --------------------------------------------------------------- builders


def _cap(cid: str, *, side_effects: list[str] | None = None, desc: str = "") -> ToolCapability:
    return ToolCapability(
        id=cid,
        package=cid.split(".")[0],
        name=cid.rsplit(".", 1)[-1],
        description=desc or cid,
        input_schema={"type": "object"},
        output_schema={},
        side_effects=side_effects or [],
    )


def _board(*tasks: BoardTask) -> TaskBoard:
    return TaskBoard(tasks=tuple(tasks))


def _task(task_id: str, name: str) -> BoardTask:
    return BoardTask(id=task_id, name=name, acceptance=("done",))


class TestNoRegistry:
    def test_all_tasks_unknown_when_registry_is_none(self) -> None:
        board = _board(_task("t1", "build"), _task("t2", "solvate"))
        result = PlanReachabilityProbe().annotate(board, None)

        assert result is not board
        for task in result.tasks:
            assert task.feasibility is not None
            assert task.feasibility.reachable is False
            assert task.feasibility.difficulty == Difficulty.UNKNOWN
            assert task.feasibility.probed_refs == ()
            assert task.feasibility.rationale == "no capability registry grounded"


class TestSearchMiss:
    def test_no_hits_marks_hard_and_unreachable(self) -> None:
        registry = InMemoryCapabilityRegistry([_cap("molpy.other.run", desc="unrelated")])
        board = _board(_task("t1", "quantum_teleport"))

        [annotated] = PlanReachabilityProbe().annotate(board, registry).tasks

        assert annotated.feasibility is not None
        assert annotated.feasibility.reachable is False
        assert annotated.feasibility.difficulty == Difficulty.HARD
        assert annotated.feasibility.probed_refs == ()


class TestSearchHit:
    def test_side_effect_free_hit_is_trivial(self) -> None:
        registry = InMemoryCapabilityRegistry(
            [_cap("molpy.build.write", desc="build a polymer")]
        )
        board = _board(_task("t1", "build"))

        [annotated] = PlanReachabilityProbe().annotate(board, registry).tasks

        assert annotated.feasibility is not None
        assert annotated.feasibility.reachable is True
        assert annotated.feasibility.difficulty == Difficulty.TRIVIAL
        assert annotated.feasibility.probed_refs == ("molpy.build.write",)
        assert "molpy.build.write" in annotated.feasibility.rationale

    def test_side_effecting_hit_is_moderate(self) -> None:
        registry = InMemoryCapabilityRegistry(
            [_cap("molpy.build.sim", desc="build", side_effects=["writes_trajectory"])]
        )
        board = _board(_task("t1", "build"))

        [annotated] = PlanReachabilityProbe().annotate(board, registry).tasks

        assert annotated.feasibility is not None
        assert annotated.feasibility.reachable is True
        assert annotated.feasibility.difficulty == Difficulty.MODERATE

    def test_probed_refs_truncated_to_max_refs_in_registry_order(self) -> None:
        registry = InMemoryCapabilityRegistry(
            [
                _cap("molpy.build.a", desc="build"),
                _cap("molpy.build.b", desc="build"),
                _cap("molpy.build.c", desc="build"),
            ]
        )
        board = _board(_task("t1", "build"))

        [annotated] = PlanReachabilityProbe(max_refs=2).annotate(board, registry).tasks

        assert annotated.feasibility is not None
        assert annotated.feasibility.probed_refs == ("molpy.build.a", "molpy.build.b")


class TestImmutabilityAndPurity:
    def test_input_board_unchanged_field_by_field(self) -> None:
        registry = InMemoryCapabilityRegistry([_cap("molpy.build.write", desc="build")])
        original_task = _task("t1", "build")
        board = _board(original_task)

        result = PlanReachabilityProbe().annotate(board, registry)

        # Input board and its task are untouched.
        assert result is not board
        assert board.version == 0
        assert board.tasks[0] is original_task
        assert board.tasks[0].feasibility is None
        assert board.tasks[0].status == TaskStatus.PENDING
        # Result is a fresh board whose task now carries the probe verdict.
        assert isinstance(result, TaskBoard)
        assert result.tasks[0].feasibility is not None

    def test_annotate_takes_no_store_parameter(self) -> None:
        params = inspect.signature(PlanReachabilityProbe.annotate).parameters
        assert list(params) == ["self", "board", "registry"]
        assert "store" not in params

    def test_annotate_returns_a_task_board_not_an_artifact(self) -> None:
        registry = InMemoryCapabilityRegistry([_cap("molpy.build.write", desc="build")])
        result = PlanReachabilityProbe().annotate(_board(_task("t1", "build")), registry)
        assert isinstance(result, TaskBoard)
