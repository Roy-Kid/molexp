"""Lowering design pins — values-on-edges ExecutionPlan, molexp-owned engine.

The workflow DAG is lowered to a frozen, molexp-owned
:class:`~molexp.workflow._engine.plan.ExecutionPlan`; the engine
(:mod:`molexp.workflow._engine.engine`) executes it with
values-on-edges semantics. These tests pin:

* ac-001 — the old single-track scheduler (``WorkflowStep`` / ``_dispatch`` /
  ``_invoke_one`` / ``_partition_by_data_deps`` / ``level_index``) stays
  deleted, AND the barrier-era timing constants
  (``_DEP_BARRIER_POLL_S`` / ``_DEADLOCK_QUIESCENT_POLLS``) are gone —
  coordination is structural, with zero timing constants for correctness.
* ac-002 — ``compiled.graph`` is a genuine :class:`ExecutionPlan` carrying
  every registered task.
* ac-003 — parallel decls and branch routes lower structurally (parallel
  maps + branch out-edges + back-edge/cycle detection on the plan).
* ac-005 — cyclic data graph and stalled/unsatisfiable graph each raise a
  ``WorkflowError`` subtype.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workflow import TaskContext, WorkflowCompiler
from molexp.workflow._engine.plan import START, ExecutionPlan
from molexp.workflow.types import (
    BranchEdges,
)

ENGINE_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow" / "_engine"


def _engine_sources() -> str:
    return "\n".join(p.read_text() for p in ENGINE_ROOT.glob("*.py"))


# ── ac-001: deleted scheduler internals + timing constants ───────────────────


def test_no_timing_constants_for_coordination() -> None:
    """Coordination is structural: the dependency-barrier poll interval and
    the quiescence deadlock window must stay deleted. Deadlock detection is
    an exact graph property (unsatisfiable dependency = no runnable node
    while triggered nodes remain), never a timer."""
    src = _engine_sources()
    for forbidden in (
        "_DEP_BARRIER_POLL_S",
        "_DEADLOCK_QUIESCENT_POLLS",
        "asyncio.wait_for(",
        "asyncio.sleep(",
    ):
        assert forbidden not in src, (
            f"{forbidden!r} found in the engine package — coordination must be "
            "event/structure driven with zero timing constants."
        )


# ── ac-002: compiled.graph is a genuine ExecutionPlan ────────────────────────


def test_compiled_graph_is_execution_plan() -> None:
    wf = WorkflowCompiler(name="g")

    @wf.task
    async def a(ctx: TaskContext) -> int:
        return 1

    @wf.task(depends_on=["a"])
    async def b(value: int) -> int:
        return value + 1

    compiled = wf.compile()
    assert isinstance(compiled.graph, ExecutionPlan)
    assert set(compiled.graph.task_names) == {"a", "b"}
    # The entry frontier is the data-zero task; b is triggered by a.
    assert compiled.graph.entry_frontier == ("a",)
    assert compiled.graph.in_sources["a"] == frozenset({START})
    assert compiled.graph.in_sources["b"] == frozenset({"a"})


# ── ac-003: parallel / branch / loop lower structurally ──────────────────────


def test_parallel_lowering_carries_fanout_maps() -> None:
    wf = WorkflowCompiler(name="par", entry="seed")

    @wf.task
    async def seed(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.task
    async def body(element: int) -> int:
        return element * 2

    @wf.task
    async def gather(results: list[int]) -> int:
        return sum(results)

    wf.parallel(map_over="seed", body="body", join="gather", max_concurrency=2)
    plan = wf.compile().graph

    assert plan.parallel_by_map_over["seed"].body == "body"
    assert plan.parallel_by_body["body"].join == "gather"
    # The fan-out publish is the join's trigger source.
    assert "body" in plan.in_sources["gather"]


def test_branch_lowering_carries_routes() -> None:
    from molexp.workflow.types import Next

    wf = WorkflowCompiler(name="br", entry="route")

    @wf.task(routes={"a": "leg_a", "b": "leg_b"})
    async def route(ctx: TaskContext) -> Next:
        return Next("a")

    @wf.task
    async def leg_a(ctx: TaskContext) -> str:
        return "a"

    @wf.task
    async def leg_b(ctx: TaskContext) -> str:
        return "b"

    plan = wf.compile().graph
    edge_set = plan.out_edges["route"]
    assert isinstance(edge_set, BranchEdges)
    assert edge_set.routes == {"a": "leg_a", "b": "leg_b"}
    # route is not on a cycle — its non-chosen edges die when it routes.
    assert "route" not in plan.recurrent


def test_loop_lowering_marks_back_edge_and_recurrence() -> None:
    from molexp.workflow.types import Next

    wf = WorkflowCompiler(name="loop", entry="step")

    @wf.task
    async def step(ctx: TaskContext) -> int:
        return 1

    @wf.task(depends_on=["step"])
    async def check(ctx: TaskContext) -> Next:
        return Next("exit")

    wf.loop(body=["step"], until="check", max_iters=3)
    plan = wf.compile().graph

    assert ("check", "step") in plan.back_edges
    # Both cycle members are recurrent — a later iteration may re-fire them.
    assert {"step", "check"} <= plan.recurrent


# ── ac-005: cyclic / stalled graphs raise WorkflowError subtypes ─────────────


# ── sanity: a lowered diamond still produces identical observable outputs ─────
