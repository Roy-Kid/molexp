"""The three equivalent ways to define a workflow task.

Matches ``docs/guide/task-and-actor.md``.

Demonstrates:

1. Decorator style — ``@wf.task`` decorates async functions.
2. OOP style — subclass ``Task`` and add via ``Workflow.add``.
3. Protocol form — any object with ``async def execute(self, ctx)``.

``Actor`` (streaming) is discussed in the guide but is only useful under a
runtime that interleaves coroutines through a ``RunContext``; see the
"Runtime Boundaries" section of the guide for what is and is not wired up
today.

Run directly::

    python examples/workflow/task_and_actor.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import Task, TaskContext, Workflow


# ── 1. Decorator style ─────────────────────────────────────────────────────
async def functional_demo() -> None:
    wf = Workflow(name="functional")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.task(depends_on=["load"])
    async def square(ctx: TaskContext) -> list[int]:
        return [x * x for x in ctx.inputs]

    result = await wf.build().execute()
    print(f"functional: {result.outputs}")


# ── 2. OOP style ───────────────────────────────────────────────────────────
class Load(Task):
    async def execute(self, ctx: TaskContext) -> list[int]:
        return [10, 20, 30]


class Sum(Task):
    async def execute(self, ctx: TaskContext) -> int:
        return sum(ctx.inputs)


async def oop_demo() -> None:
    spec = (
        Workflow(name="oop")
        .add(Load())
        .add(Sum(), depends_on=["load"])
        .build()
    )
    result = await spec.execute()
    print(f"oop:        {result.outputs}")


# ── 3. Protocol form — third-party object, no molexp import needed ─────────
class ExternalDoubler:
    """Matches :class:`~molexp.workflow.protocols.Runnable` structurally."""

    async def execute(self, ctx) -> int:
        return sum(ctx.inputs) * 2


async def protocol_demo() -> None:
    spec = (
        Workflow(name="external")
        .add(Load())
        .add(ExternalDoubler(), name="double", depends_on=["load"])
        .build()
    )
    result = await spec.execute()
    print(f"protocol:   {result.outputs}")


async def main() -> None:
    await functional_demo()
    await oop_demo()
    await protocol_demo()


if __name__ == "__main__":
    asyncio.run(main())
