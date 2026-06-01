"""Regression test: ``_GRAPH.run`` must receive ``inputs=WorkflowStep(...)``.

The bug we hit: pydantic-graph 1.x requires the start node to be passed
explicitly via ``inputs=``. Three call sites in
``molexp.workflow._pydantic_graph.runtime`` used to call ``_GRAPH.run``
without it, so every workflow execution exploded with
``ValueError: Node None is not of type WorkflowStep`` mid-run.

All three call sites now route through ``_run_graph`` /
``_iter_graph``; this test makes sure that survives.
"""

from __future__ import annotations

import asyncio

import molexp as me
from molexp.workflow import TaskContext, WorkflowBuilder


def test_single_task_workflow_executes(tmp_path):
    builder = WorkflowBuilder(name="oneshot")

    @builder.task
    async def hello(ctx: TaskContext) -> dict:
        return {"ok": True}

    spec = builder.build()

    ws = me.Workspace(tmp_path)
    proj = ws.add_project("p")
    exp = proj.add_experiment("e", params={})
    spec.bind_to(exp)
    run = exp.add_run({})

    from molexp.workspace.run import RunContext

    with RunContext(run) as ctx:
        result = asyncio.run(spec.execute(run_context=ctx))

    assert result.status == "completed", (
        f"workflow execute regressed — likely missing inputs=WorkflowStep in _GRAPH.run; "
        f"got result={result!r}"
    )
    assert result.outputs == {"hello": {"ok": True}}


def test_two_task_chain_workflow_executes(tmp_path):
    """A two-task chain — exercises ``WorkflowStep(level_index=N+1)`` return path too."""
    builder = WorkflowBuilder(name="chain")

    @builder.task
    async def first(ctx: TaskContext) -> int:
        return 1

    @builder.task(depends_on=["first"])
    async def second(ctx: TaskContext) -> int:
        return ctx.inputs + 1

    spec = builder.build()

    ws = me.Workspace(tmp_path)
    proj = ws.add_project("p")
    exp = proj.add_experiment("e", params={})
    spec.bind_to(exp)
    run = exp.add_run({})

    from molexp.workspace.run import RunContext

    with RunContext(run) as ctx:
        result = asyncio.run(spec.execute(run_context=ctx))

    assert result.status == "completed"
    assert result.outputs == {"first": 1, "second": 2}
