"""Routed branches and workflow-level loops — values ride the edges.

Matches ``docs/guide/control-flow.md``.

``wf.branch`` declares label-routed edges: the deciding task returns
``(value, Next("label"))`` and the routed target receives ``value`` bound to its
own named parameters. ``wf.loop`` repeats a body until its ``until`` task returns
``Next("exit")``; each iteration's value reaches the next iteration's body head
as a named parameter — no shared mutable state, no ``ctx``. This example runs
two patterns back to back:

1. Branch — ``classify`` routes a payload to ``accepted`` or ``rejected``.
2. Loop — ``step`` / ``check`` refine a value until it converges, then
   ``report`` receives the final value on the exit edge.

Run directly::

    python examples/workflow/branch_and_loop.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import Next, WorkflowCompiler, WorkflowRuntime


# ── 1. Branch — route a value to one downstream task ───────────────────────
async def branch_demo(score: float) -> None:
    wf = WorkflowCompiler(name="triage", entry="classify")

    @wf.task
    async def classify() -> tuple[dict, Next]:
        label = "accept" if score > 0.5 else "reject"
        return {"score": score}, Next(label)

    @wf.task
    async def accepted(score: float) -> str:
        return f"accepted score={score}"  # routed dict binds by name

    @wf.task
    async def rejected(score: float) -> str:
        return f"rejected score={score}"

    wf.branch("classify", routes={"accept": "accepted", "reject": "rejected"})

    result = await WorkflowRuntime().execute(wf.compile())
    taken = "accepted" if "accepted" in result.outputs else "rejected"
    print(f"branch: {result.outputs[taken]}")


# ── 2. Loop — repeat the body until the until-task exits ───────────────────
async def loop_demo() -> None:
    wf = WorkflowCompiler(name="refine", entry="step")

    @wf.task
    async def step(value: int | None = None) -> int:
        # ``value`` = previous iteration's routed output (None on iteration 1).
        prev = value if isinstance(value, int) else 0
        return prev + 1

    @wf.task(depends_on=["step"])
    async def check(value: int) -> tuple[int, Next]:
        # The single upstream (``step``) value binds positionally to ``value``.
        return value, Next("exit" if value >= 3 else "continue")

    @wf.task
    async def report(value: int) -> str:
        return f"converged at {value}"  # exit-edge value binds by name

    wf.loop(body=["step"], until="check", max_iters=10, on_exit="report")

    result = await WorkflowRuntime().execute(wf.compile())
    print(f"loop:   {result.outputs['report']} (step ran {result.outputs['step']} times)")


async def main() -> None:
    await branch_demo(score=0.9)
    await branch_demo(score=0.2)
    await loop_demo()


if __name__ == "__main__":
    asyncio.run(main())
