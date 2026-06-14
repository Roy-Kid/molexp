"""The three equivalent ways to define a workflow task, plus a streaming actor.

Matches ``docs/guide/task-and-actor.md``.

Demonstrates:

1. Decorator style — ``@wf.task`` decorates async functions.
2. OOP style — subclass ``Task`` and register via ``WorkflowCompiler.add``.
3. Protocol form — any object with ``async def execute(self, ctx)``.
4. Actor — ``@wf.actor`` (or ``Actor`` subclass) defines an async generator;
   the engine drives it to exhaustion and records the last yielded value.

Run directly::

    python examples/workflow/task_and_actor.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import Task, TaskContext, WorkflowCompiler, WorkflowRuntime


# ── 1. Decorator style ─────────────────────────────────────────────────────
async def functional_demo() -> None:
    wf = WorkflowCompiler(name="functional")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.task(depends_on=["load"])
    async def square(ctx: TaskContext) -> list[int]:
        return [x * x for x in ctx.inputs]

    result = await WorkflowRuntime().execute(wf.compile())
    print(f"functional: {result.outputs}")


# ── 2. OOP style ───────────────────────────────────────────────────────────
class Load(Task):
    async def execute(self, ctx: TaskContext) -> list[int]:
        return [10, 20, 30]


class Sum(Task):
    async def execute(self, ctx: TaskContext) -> int:
        return sum(ctx.inputs)


async def oop_demo() -> None:
    compiled = WorkflowCompiler(name="oop").add(Load()).add(Sum(), depends_on=["load"]).compile()
    result = await WorkflowRuntime().execute(compiled)
    print(f"oop:        {result.outputs}")


# ── 3. Protocol form — third-party object, no molexp import needed ─────────
class ExternalDoubler:
    """Matches :class:`~molexp.workflow.protocols.Runnable` structurally."""

    async def execute(self, ctx) -> int:
        return sum(ctx.inputs) * 2


async def protocol_demo() -> None:
    compiled = (
        WorkflowCompiler(name="external")
        .add(Load())
        .add(ExternalDoubler(), name="double", depends_on=["load"])
        .compile()
    )
    result = await WorkflowRuntime().execute(compiled)
    print(f"protocol:   {result.outputs}")


# ── 4. Streaming actor — async generator driven to exhaustion ──────────────
async def actor_demo() -> None:
    wf = WorkflowCompiler(name="stream")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.actor(depends_on=["load"])
    async def monitor(ctx: TaskContext):
        for item in ctx.inputs:
            yield {"seen": item}  # last yield becomes the task output

    result = await WorkflowRuntime().execute(wf.compile())
    print(f"actor:      {result.outputs}")


async def main() -> None:
    await functional_demo()
    await oop_demo()
    await protocol_demo()
    await actor_demo()


if __name__ == "__main__":
    asyncio.run(main())
