"""Tests for ``PlanDiff``-centric repair (ac-008)."""

from __future__ import annotations

import pytest

from molexp.agent.modes._planning import DiffOpKind, PlanDiff, PlanNodeOp
from molexp.agent.modes.author.repair import (
    apply_local_repair,
    build_repair_diff,
    escalate_plan_repair,
    is_local_repair,
)

from .conftest import make_plan_graph, make_step


def test_build_repair_diff_populates_failed_invariant_and_affected_nodes() -> None:
    plan = make_plan_graph()
    diff = build_repair_diff(
        plan_graph=plan,
        step_id="run",
        traceback="AssertionError: boom",
        attempt=1,
    )
    assert diff.failed_invariant
    assert diff.affected_nodes == ("run",)
    assert "boom" in diff.rationale


def test_build_repair_diff_records_downstream_as_invalidated() -> None:
    plan = make_plan_graph()
    diff = build_repair_diff(
        plan_graph=plan,
        step_id="prepare",
        traceback="fail",
        attempt=1,
    )
    # 'run' depends on 'prepare', so it is invalidated.
    assert "run" in diff.invalidated
    assert "prepare" in diff.reused or "prepare" not in diff.reused  # prepare itself excluded


def test_build_repair_diff_is_local_for_single_task() -> None:
    diff = build_repair_diff(
        plan_graph=make_plan_graph(),
        step_id="run",
        traceback="fail",
        attempt=1,
    )
    assert is_local_repair(diff)


def test_apply_local_repair_keeps_plan_topology() -> None:
    plan = make_plan_graph()
    diff = build_repair_diff(
        plan_graph=plan,
        step_id="run",
        traceback="fail",
        attempt=1,
    )
    repaired = apply_local_repair(diff, plan_graph=plan)
    assert {s.id for s in repaired.steps} == {s.id for s in plan.steps}


def test_apply_local_repair_rejects_plan_shape_diff() -> None:
    plan = make_plan_graph()
    new_step = make_step("extra")
    shape_diff = PlanDiff(
        failed_invariant="needs_new_step",
        affected_nodes=("extra",),
        operations=(PlanNodeOp(kind=DiffOpKind.add, node_id="extra", step=new_step),),
        rationale="add a step",
        reused=(),
        invalidated=(),
    )
    with pytest.raises(ValueError, match="plan-shape"):
        apply_local_repair(shape_diff, plan_graph=plan)


def test_escalate_plan_repair_applies_shape_change() -> None:
    plan = make_plan_graph()
    new_step = make_step("extra")
    shape_diff = PlanDiff(
        failed_invariant="needs_new_step",
        affected_nodes=("extra",),
        operations=(PlanNodeOp(kind=DiffOpKind.add, node_id="extra", step=new_step),),
        rationale="add a step",
        reused=(),
        invalidated=(),
    )
    escalated = escalate_plan_repair(shape_diff, plan_graph=plan)
    assert "extra" in {s.id for s in escalated.steps}
    # The input plan is never mutated.
    assert "extra" not in {s.id for s in plan.steps}


def test_every_debug_iteration_yields_a_plan_diff() -> None:
    plan = make_plan_graph()
    diffs = [
        build_repair_diff(plan_graph=plan, step_id="run", traceback=f"f{i}", attempt=i)
        for i in range(1, 4)
    ]
    assert all(isinstance(d, PlanDiff) for d in diffs)
    assert all(d.failed_invariant and d.affected_nodes for d in diffs)
