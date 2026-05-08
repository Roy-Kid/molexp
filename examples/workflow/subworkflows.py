"""Compose an outer workflow out of a reusable inner ``WorkflowSpec``.

Matches ``docs/guide/subworkflows.md``.

There is no dedicated ``SubWorkflow`` type. You wrap the inner spec in a
``Task`` and let the outer workflow treat it as one opaque node. The
inner spec retains its own ``workflow_id``, topology, and validation.

Run directly::

    python examples/workflow/subworkflows.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import Task, TaskContext, Workflow, WorkflowSpec


def build_preprocess() -> WorkflowSpec:
    """The inner pipeline — could live in its own module."""
    wf = Workflow(name="preprocess")

    @wf.task
    async def load(ctx: TaskContext) -> list[float]:
        return [3.0, 1.0, 4.0, 1.0, 5.0]

    @wf.task(depends_on=["load"])
    async def normalize(ctx: TaskContext) -> list[float]:
        top = max(ctx.inputs)
        return [x / top for x in ctx.inputs]

    return wf.build()


class Preprocess(Task):
    """Wraps an inner spec so an outer builder can add it as one node."""

    def __init__(self, spec: WorkflowSpec) -> None:
        self._spec = spec

    async def execute(self, ctx: TaskContext) -> list[float]:
        # Forward the outer ``run_context`` (when present) so inner-task
        # workspace helpers continue to work; otherwise the inner spec
        # runs with no workspace attached, just like the outer when you
        # call ``spec.execute()`` from a notebook.
        result = await self._spec.execute(run_context=ctx.run_context)
        return result.outputs["normalize"]


class Train(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return sum(ctx.inputs) / len(ctx.inputs)


async def main() -> None:
    inner = build_preprocess()

    outer = (
        Workflow(name="train")
        .add(Preprocess(inner), name="preprocess")
        .add(Train(), depends_on=["preprocess"])
        .build()
    )

    result = await outer.execute()
    print(f"inner workflow_id: {inner.workflow_id}")
    print(f"outer workflow_id: {outer.workflow_id}")
    print(f"status:            {result.status}")
    print(f"outputs:           {result.outputs}")


if __name__ == "__main__":
    asyncio.run(main())
