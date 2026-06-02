"""Spec workflow-refactor-03-pg-node-lowering — genuine per-task pg lowering.

The workflow DAG is lowered to a real ``pydantic_graph`` Graph with one
``Step`` per task; control flow rides pg primitives (edges / Join / Fork /
Decision / reduce_* reducers). These tests cover:

* ac-001 — the old single-track scheduler (``WorkflowStep`` / ``_dispatch`` /
  ``_invoke_one`` / ``_partition_by_data_deps`` / ``level_index``) is deleted;
  ``GraphBuilder`` / ``Join`` reducers / ``Decision`` are imported + used.
* ac-002 — ``compiled.graph`` is a genuine ``pydantic_graph.graph_builder.Graph``.
* ac-003 — parallel → Fork+Join, branch → Decision, reduce_* reducers used.
* ac-005 — cyclic data graph and stalled/unsatisfiable graph each raise a
  ``WorkflowError`` subtype.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_graph.graph_builder import Graph

from molexp.workflow import GraphWorkflowRuntime, TaskContext, WorkflowCompiler
from molexp.workflow.types import CycleError, UnreachableTaskError, WorkflowError

PG_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow" / "_pydantic_graph"


def _pg_sources() -> str:
    return "\n".join(p.read_text() for p in PG_ROOT.glob("*.py"))


# ── ac-001: deleted scheduler internals ──────────────────────────────────────


def test_old_scheduler_internals_deleted() -> None:
    src = _pg_sources()
    for forbidden in (
        "class WorkflowStep",
        "_dispatch",
        "_invoke_one",
        "_partition_by_data_deps",
        "level_index",
    ):
        assert forbidden not in src, (
            f"{forbidden!r} must be deleted — the single-track scheduler is replaced "
            "by genuine per-task pg lowering."
        )


def test_pg_primitives_imported_and_used() -> None:
    compiler_src = (PG_ROOT / "compiler.py").read_text()
    assert "GraphBuilder" in compiler_src
    assert "gb.decision(" in compiler_src
    assert "gb.join(" in compiler_src
    assert "reduce_null" in compiler_src
    assert "reduce_dict_update" in compiler_src
    assert ".map(" in compiler_src  # map-Fork fan-out for wf.parallel


# ── ac-002: compiled.graph is a genuine pg Graph ─────────────────────────────


def test_compiled_graph_is_pydantic_graph_graph() -> None:
    wf = WorkflowCompiler(name="g")

    @wf.task
    async def a(ctx: TaskContext) -> int:
        return 1

    @wf.task(depends_on=["a"])
    async def b(ctx: TaskContext) -> int:
        return ctx.inputs + 1

    compiled = wf.compile()
    assert isinstance(compiled.graph, Graph)
    # Node count relates to task count: a Step per task (plus routing nodes).
    node_ids = set(compiled.graph.nodes)
    assert {"a", "b"} <= node_ids


# ── ac-003: parallel → Fork+Join, branch → Decision ──────────────────────────


def _graph_nodes(compiled) -> dict[str, object]:
    return dict(compiled.graph.nodes)


def test_parallel_graph_has_fork_and_join() -> None:
    from pydantic_graph.join import Join

    wf = WorkflowCompiler(name="par", entry="seed")

    @wf.task
    async def seed(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.task
    async def body(ctx: TaskContext) -> int:
        return ctx.inputs * 2

    @wf.task
    async def gather(ctx: TaskContext) -> int:
        return sum(ctx.inputs)

    wf.parallel(map_over="seed", body="body", join="gather", max_concurrency=2)
    compiled = wf.compile()

    nodes = _graph_nodes(compiled)
    # A reduce_dict_update Join collects the fan-out.
    assert any(isinstance(n, Join) for n in nodes.values()), "wf.parallel must lower to a Join node"
    # The render carries a fork (map fan-out from start/seed).
    rendered = compiled.graph.render()
    assert "fork" in rendered.lower()


def test_branch_graph_has_decision() -> None:
    from pydantic_graph.decision import Decision

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

    compiled = wf.compile()
    nodes = _graph_nodes(compiled)
    assert any(isinstance(n, Decision) for n in nodes.values()), (
        "wf.branch must lower to a Decision node"
    )


def test_reduce_reducers_used_in_lowering() -> None:
    """The Join reducers used are the genuine pydantic-graph ``reduce_*``."""
    from pydantic_graph.join import reduce_dict_update, reduce_null

    # Imported names exist and are callables — the compiler references them.
    assert callable(reduce_null)
    assert callable(reduce_dict_update)
    compiler_src = (PG_ROOT / "compiler.py").read_text()
    assert "reduce_null" in compiler_src
    assert "reduce_dict_update" in compiler_src


# ── ac-005: cyclic / stalled graphs raise WorkflowError subtypes ─────────────


def test_cyclic_data_graph_raises_workflow_error() -> None:
    wf = WorkflowCompiler(name="cyc")

    @wf.task(depends_on=["b"])
    async def a(ctx: TaskContext) -> int:
        return 1

    @wf.task(depends_on=["a"])
    async def b(ctx: TaskContext) -> int:
        return 2

    with pytest.raises(WorkflowError) as exc:
        wf.compile()
    assert isinstance(exc.value, CycleError)


def test_unreachable_task_raises_workflow_error() -> None:
    wf = WorkflowCompiler(name="stall", entry="a")

    @wf.task
    async def a(ctx: TaskContext) -> int:
        return 1

    # `orphan` has no incoming edge from the entry frontier → unreachable.
    @wf.task
    async def orphan(ctx: TaskContext) -> int:
        return 2

    with pytest.raises(WorkflowError) as exc:
        wf.compile()
    assert isinstance(exc.value, UnreachableTaskError)


# ── sanity: a lowered diamond still produces identical observable outputs ─────


def test_diamond_outputs_preserved() -> None:
    wf = WorkflowCompiler(name="diamond")

    @wf.task
    async def root(ctx: TaskContext) -> int:
        return 1

    @wf.task(depends_on=["root"])
    async def left(ctx: TaskContext) -> int:
        return ctx.inputs + 10

    @wf.task(depends_on=["root"])
    async def right(ctx: TaskContext) -> int:
        return ctx.inputs + 20

    @wf.task(depends_on=["left", "right"])
    async def sink(ctx: TaskContext) -> int:
        return ctx.inputs["left"] + ctx.inputs["right"]

    result = asyncio.run(GraphWorkflowRuntime().execute(wf.compile()))
    assert result.status == "completed"
    assert result.outputs == {"root": 1, "left": 11, "right": 21, "sink": 32}
