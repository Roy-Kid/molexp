"""A ``Workflow`` executed with no workspace attached.

Matches ``docs/getting-started/first-workflow.md``.

The point of this example is that a workflow is independent of the workspace
model: you can build and execute one entirely in memory. Persistent runs,
artifacts, profiles, and catalogs are all additive layers added later.

Run directly::

    python examples/getting_started/02_first_workflow.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import TaskContext, Workflow, WorkflowBuilder


async def main() -> None:
    wf = WorkflowBuilder(name="first-workflow")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3, 4, 5]

    @wf.task(depends_on=["load"])
    async def square(ctx: TaskContext) -> list[int]:
        return [x * x for x in ctx.inputs]

    @wf.task(depends_on=["square"])
    async def total(ctx: TaskContext) -> int:
        return sum(ctx.inputs)

    spec = wf.build()
    result = await spec.execute()

    print(f"workflow_id: {spec.workflow_id}")
    print(f"status:      {result.status}")
    print(f"outputs:     {result.outputs}")


if __name__ == "__main__":
    asyncio.run(main())
