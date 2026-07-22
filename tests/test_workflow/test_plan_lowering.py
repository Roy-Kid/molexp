"""ExecutionPlan lowering — values-on-edges plan over the validated topology.

The workflow DAG is lowered to a frozen, molexp-owned
:class:`~molexp.workflow._engine.plan.ExecutionPlan`; the engine
(:mod:`molexp.workflow._engine.engine`) executes it with values-on-edges
semantics. These tests pin the *structure* the compiler lowers to (the plan
fields the engine reads), and the invariant that coordination carries zero
timing constants:

* the barrier-era timing constants are gone — coordination is structural,
  with no timing constant used for correctness (deadlock is an exact graph
  property, not a timer);
* ``compiled.graph`` is a genuine :class:`ExecutionPlan` carrying every
  registered task, with the entry frontier and forward in-sources lowered;
* ``wf.parallel`` / branch routes / ``wf.loop`` lower structurally (fan-out
  maps, branch out-edges, and back-edge/recurrence marking on the plan).

Execution *behavior* of parallel / branch / loop is owned by their own suites
(``test_parallel`` / ``test_routes`` / ``test_values_on_edges`` /
``test_deadlock_guard``); here we assert only the lowered plan shape.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workflow import TaskContext, WorkflowCompiler
from molexp.workflow._engine.plan import START, ExecutionPlan
from molexp.workflow.types import BranchEdges, Next

ENGINE_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow" / "_engine"


def _engine_sources() -> str:
    return "\n".join(p.read_text() for p in ENGINE_ROOT.glob("*.py"))


def test_no_timing_constants_for_coordination() -> None:
    """Invariant lock: coordination is structural. The dependency-barrier poll
    interval and the quiescence deadlock window must stay deleted, and the
    engine must not fall back on ``asyncio.sleep`` / ``asyncio.wait_for`` for
    correctness — deadlock is an exact graph property (unsatisfiable dependency
    = no runnable node while triggered nodes remain), never a timer."""
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


class TestExecutionPlan:
    """The lowered ``ExecutionPlan`` structure carried by ``compiled.graph``."""

    def test_compiled_graph_is_execution_plan(self) -> None:
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

    def test_parallel_lowering_carries_fanout_maps(self) -> None:
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

    def test_branch_lowering_carries_routes(self) -> None:
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

    def test_loop_lowering_marks_back_edge_and_recurrence(self) -> None:
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
