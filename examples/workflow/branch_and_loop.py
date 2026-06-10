"""Routed branches and workflow-level loops — values ride the edges.

Matches ``docs/guide/control-flow.md``.

``wf.branch`` declares label-routed edges: the deciding task returns
``(value, Next("label"))`` and the routed target receives ``value`` as its
``ctx.inputs``. ``wf.loop`` repeats a body until its ``until`` task returns
``Next("exit")``; each iteration's value reaches the next iteration's body
head via ``ctx.inputs`` — no shared mutable state. This example runs two
patterns back to back:

1. Branch — ``classify`` routes a payload to ``accepted`` or ``rejected``.
2. Loop — ``step`` / ``check`` refine a value until it converges, then
   ``report`` receives the final value on the exit edge.

Run directly::

    python examples/workflow/branch_and_loop.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import Next, TaskContext, WorkflowCompiler, WorkflowRuntime


# ── 1. Branch — route a value to one downstream task ───────────────────────
async def branch_demo(score: float) -> None:
    wf = WorkflowCompiler(name="triage", entry="classify")

    @wf.task
    async def classify(ctx: TaskContext) -> tuple[dict, Next]:
        label = "accept" if score > 0.5 else "reject"
        return {"score": score}, Next(label)

    @wf.task
    async def accepted(ctx: TaskContext) -> str:
        return f"accepted score={ctx.inputs['score']}"  # routed value via ctx.inputs

    @wf.task
    async def rejected(ctx: TaskContext) -> str:
        return f"rejected score={ctx.inputs['score']}"

    wf.branch("classify", routes={"accept": "accepted", "reject": "rejected"})

    result = await WorkflowRuntime().execute(wf.compile())
    taken = "accepted" if "accepted" in result.outputs else "rejected"
    print(f"branch: {result.outputs[taken]}")


# ── 2. Loop — repeat the body until the until-task exits ───────────────────
async def loop_demo() -> None:
    wf = WorkflowCompiler(name="refine", entry="step")

    @wf.task
    async def step(ctx: TaskContext) -> int:
        prev = ctx.inputs if isinstance(ctx.inputs, int) else 0
        return prev + 1  # ctx.inputs = previous iteration's value (None on iter 1)

    @wf.task(depends_on=["step"])
    async def check(ctx: TaskContext) -> tuple[int, Next]:
        n = ctx.inputs
        return n, Next("exit" if n >= 3 else "continue")

    @wf.task
    async def report(ctx: TaskContext) -> str:
        return f"converged at {ctx.inputs}"  # exit-edge value via ctx.inputs

    wf.loop(body=["step"], until="check", max_iters=10, on_exit="report")

    result = await WorkflowRuntime().execute(wf.compile())
    print(f"loop:   {result.outputs['report']} (step ran {result.outputs['step']} times)")


async def main() -> None:
    await branch_demo(score=0.9)
    await branch_demo(score=0.2)
    await loop_demo()


if __name__ == "__main__":
    asyncio.run(main())
