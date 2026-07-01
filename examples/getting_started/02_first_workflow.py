"""A workflow executed with no workspace attached.

Matches ``docs/getting-started/first-workflow.md``.

The point of this example is that a workflow is independent of the workspace
model: you can compile and execute one entirely in memory. Persistent runs,
artifacts, profiles, and catalogs are all additive layers added later.

Run directly::

    python examples/getting_started/02_first_workflow.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import WorkflowCompiler, WorkflowRuntime


async def main() -> None:
    wf = WorkflowCompiler(name="first-workflow")

    @wf.task
    async def load() -> list[int]:
        return [1, 2, 3, 4, 5]

    # A downstream task receives an upstream's output by naming a parameter after
    # that upstream task — ``square`` binds ``load``'s output to its ``load`` arg.
    @wf.task(depends_on=["load"])
    async def square(load: list[int]) -> list[int]:
        return [x * x for x in load]

    @wf.task(depends_on=["square"])
    async def total(square: list[int]) -> int:
        return sum(square)

    compiled = wf.compile()
    result = await WorkflowRuntime().execute(compiled)

    print(f"workflow_id: {compiled.workflow_id}")
    print(f"status:      {result.status}")
    print(f"outputs:     {result.outputs}")


if __name__ == "__main__":
    asyncio.run(main())
