"""Compose an outer workflow out of a reusable inner workflow via ``SubWorkflow``.

Matches ``docs/guide/subworkflows.md``.

``SubWorkflow`` is the sanctioned composition node: it wraps an inner
``CompiledWorkflow`` (or a ``WorkflowCompiler``, compiled on construction) and,
when executed, runs that inner spec end-to-end through the engine — forwarding
the outer ``run_context`` by identity. From the outer graph's perspective it is
a single registered task, so it also slots into ``builder.parallel(body=...)``
as the per-element fan-out body with no per-element node growth.

Run directly::

    python examples/workflow/subworkflows.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import (
    SubWorkflow,
    Task,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)


def build_preprocess() -> WorkflowCompiler:
    """The inner pipeline — could live in its own module.

    ``load`` reads ``ctx.inputs`` (the value forwarded into the inner workflow's
    entry task), so when this spec is used as a ``parallel`` body each fan-out
    element drives a *distinct* inner run.
    """
    wf = WorkflowCompiler(name="preprocess")

    @wf.task
    async def load(ctx: TaskContext) -> list[float]:
        # ``ctx.inputs`` is the forwarded node input: the fan-out element (an int)
        # in the parallel case, or ``None`` when this inner is run standalone.
        seed = ctx.inputs if isinstance(ctx.inputs, int) else 0
        return [3.0 + seed, 1.0, 4.0, 1.0, 5.0]

    @wf.task(depends_on=["load"])
    async def normalize(ctx: TaskContext) -> list[float]:
        top = max(ctx.inputs)
        return [x / top for x in ctx.inputs]

    return wf


class Train(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return sum(ctx.inputs) / len(ctx.inputs)


async def run_chained() -> None:
    """SubWorkflow as one node of an outer chain: preprocess → train."""
    outer = (
        WorkflowCompiler(name="train")
        .add(SubWorkflow(build_preprocess()), name="preprocess")
        .add(Train(), depends_on=["preprocess"])
        .compile()
    )

    result = await WorkflowRuntime().execute(outer)
    print("── chained ──")
    print(f"status:  {result.status}")
    print(f"outputs: {result.outputs}")


async def run_parallel_body() -> None:
    """SubWorkflow as the per-element body of ``builder.parallel``.

    The fan-out runs the full inner chain once per element; each element is
    forwarded into the inner entry task (``load``), so ``join`` receives one
    *distinct* inner output per element in iteration order. The compiled task set
    stays exactly {enumerate, preprocess, collect} — no per-element node growth.
    """
    wf = WorkflowCompiler(name="fanout", entry="enumerate")

    @wf.task
    async def enumerate(ctx: TaskContext) -> list[int]:
        return [0, 1, 2]

    wf.add(SubWorkflow(build_preprocess()), name="preprocess")

    @wf.task
    async def collect(ctx: TaskContext) -> list[list[float]]:
        return list(ctx.inputs)

    wf.parallel(map_over="enumerate", body="preprocess", join="collect", max_concurrency=2)

    compiled = wf.compile()
    result = await WorkflowRuntime().execute(compiled)
    print("── parallel body ──")
    print(f"task set: {sorted(t.name for t in compiled._tasks)}")
    print(f"status:   {result.status}")
    print(f"collect:  {result.outputs['collect']}")


async def main() -> None:
    await run_chained()
    await run_parallel_body()


if __name__ == "__main__":
    asyncio.run(main())
