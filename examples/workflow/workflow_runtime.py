"""``runtime.execute()`` vs ``runtime.start()`` — the two runtime entry points.

Matches ``docs/guide/workflow-runtime.md``.

Execution lives on ``WorkflowRuntime``, not on the compiled artifact.
``execute()`` blocks and returns a ``WorkflowResult``; ``start()`` launches
the same execution in the background and returns a ``WorkflowExecution``
handle that can be awaited or cancelled.

Run directly::

    python examples/workflow/workflow_runtime.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import CompiledWorkflow, TaskContext, WorkflowCompiler, WorkflowRuntime


def build_slow_workflow() -> CompiledWorkflow:
    wf = WorkflowCompiler(name="slow")

    @wf.task
    async def step_one(ctx: TaskContext) -> int:
        await asyncio.sleep(0.05)
        return 1

    @wf.task(depends_on=["step_one"])
    async def step_two(ctx: TaskContext) -> int:
        await asyncio.sleep(0.05)
        return ctx.inputs + 1

    return wf.compile()


async def blocking_entry() -> None:
    """``execute()`` — simplest call shape. Awaits to completion."""
    result = await WorkflowRuntime().execute(build_slow_workflow())
    print(f"execute:   status={result.status}, outputs={result.outputs}")


async def background_entry() -> None:
    """``start()`` — fire-and-observe handle; awaitable and cancellable."""
    handle = await WorkflowRuntime().start(build_slow_workflow())
    print(f"start:     launched handle={handle!r}")
    result = await handle.wait()
    print(f"           joined: status={result.status}, outputs={result.outputs}")


async def main() -> None:
    await blocking_entry()
    await background_entry()


if __name__ == "__main__":
    asyncio.run(main())
