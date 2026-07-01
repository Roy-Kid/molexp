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

from molexp.workflow import CompiledWorkflow, WorkflowCompiler, WorkflowRuntime


def build_slow_workflow() -> CompiledWorkflow:
    wf = WorkflowCompiler(name="slow")

    @wf.task
    async def step_one() -> int:
        await asyncio.sleep(0.05)
        return 1

    # ``step_two`` receives ``step_one``'s output by naming a parameter after it.
    @wf.task(depends_on=["step_one"])
    async def step_two(step_one: int) -> int:
        await asyncio.sleep(0.05)
        return step_one + 1

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
