"""Values-on-edges delivery — routed / loop-back values arrive via ``ctx.inputs``.

Precondition contract for the deferred ``pure-task-context-NN-state-elimination``
spec: when a ``wf.loop`` feeds iteration N's output into iteration N+1, or a
branch routes a ``(value, Next(label))``, the receiving task can read that
value through its ``{inputs}`` channel instead of ``ctx.state``.

Delivery rules (engine ``_launch`` + ``run_task_body``):

* the declared ``depends_on`` interface always wins — a task with data deps
  keeps its collected upstream dict/value shape;
* a dep-less task receives the value carried on its activating trigger edge
  (branch route, loop back-edge, or unconditional control edge);
* engine-injected ``root_inputs`` (workspace run params + workdir) MERGE with
  a delivered dict (delivered keys win), mirroring SubWorkflow forwarding.
"""

from __future__ import annotations

import pytest

from molexp.workflow import WorkflowCompiler, WorkflowRuntime
from molexp.workflow.types import Next

# ── loop-back: iteration N's routed value reaches iteration N+1's inputs ─────


@pytest.mark.asyncio
async def test_loop_back_value_delivered_via_inputs() -> None:
    """``wf.loop``: the until-task returns ``(value, Next("continue"))`` and
    the dep-less loop head observes that value as ``ctx.inputs`` on the next
    iteration — no ``ctx.state`` read required."""
    seen_inputs: list[object] = []

    wf = WorkflowCompiler(name="loop-values", entry="head")

    @wf.task
    async def head(ctx) -> int:
        seen_inputs.append(ctx.inputs)
        prev = ctx.inputs if isinstance(ctx.inputs, int) else 0
        return prev + 1

    @wf.task(depends_on=["head"])
    async def check(ctx) -> tuple[int, Next]:
        n = ctx.inputs
        return n, Next("exit" if n >= 3 else "continue")

    wf.loop(body=["head"], until="check", max_iters=10)

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
    # Iteration 1 has no incoming value; iterations 2 and 3 receive the
    # previous iteration's routed output through ctx.inputs.
    assert seen_inputs == [None, 1, 2]
    assert result.outputs["head"] == 3
    assert result.outputs["check"] == 3


@pytest.mark.asyncio
async def test_self_loop_value_delivered_via_inputs() -> None:
    """A self-looping branch task receives its own previous output as
    ``ctx.inputs`` on re-entry."""
    seen_inputs: list[object] = []

    wf = WorkflowCompiler(name="self-loop-values", entry="tick")

    @wf.task(routes={"again": "tick", "done": "_end"})
    async def tick(ctx) -> tuple[int, Next]:
        seen_inputs.append(ctx.inputs)
        n = (ctx.inputs or 0) + 1
        return n, Next("again" if n < 3 else "done")

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
    assert seen_inputs == [None, 1, 2]
    assert result.outputs["tick"] == 3


# ── branch routing: (value, Next(label)) reaches the routed target's inputs ──


@pytest.mark.asyncio
async def test_branch_routed_value_delivered_via_inputs() -> None:
    seen: dict[str, object] = {}

    wf = WorkflowCompiler(name="branch-values", entry="route")

    @wf.task(routes={"go": "dst", "stop": "_end"})
    async def route(ctx) -> tuple[dict, Next]:
        return {"payload": 42}, Next("go")

    @wf.task
    async def dst(ctx) -> object:
        seen["inputs"] = ctx.inputs
        return ctx.inputs

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
    assert seen["inputs"] == {"payload": 42}, (
        "a branch-routed (value, Next(label)) must arrive at the dep-less target as ctx.inputs"
    )


# ── precedence: declared depends_on beats edge-delivered values ───────────────


@pytest.mark.asyncio
async def test_declared_depends_on_wins_over_delivered_value() -> None:
    """A routed target WITH ``depends_on`` keeps the declared collection shape;
    the trigger-carried value never overrides the data interface."""
    seen: dict[str, object] = {}

    wf = WorkflowCompiler(name="deps-win", entry="src")

    @wf.task
    async def base(ctx) -> str:
        return "base-out"

    @wf.task(routes={"go": "consumer"})
    async def src(ctx) -> tuple[str, Next]:
        return "routed-out", Next("go")

    @wf.task(depends_on=["base"])
    async def consumer(ctx) -> object:
        seen["inputs"] = ctx.inputs
        return ctx.inputs

    wf.entry("base")  # both src (constructor) and base are entries

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
    assert seen["inputs"] == "base-out"


# ── root_inputs merge: loop-back into a workspace root keeps params ───────────


@pytest.mark.asyncio
async def test_loop_back_merges_with_root_inputs(tmp_path) -> None:
    """A loop head that is also a workspace ROOT task sees the engine-injected
    ``{params, workdir}`` AND the loop-delivered keys merged in (delivered
    keys win) — same merge rule as SubWorkflow root_input forwarding."""
    seen_inputs: list[object] = []

    class _RunContextStub:
        def __init__(self, work_dir):
            self.work_dir = work_dir
            self.params = {"alpha": 7}
            self.config = {}
            self.run = type("RunStub", (), {"id": "stub-run", "run_dir": work_dir})()

    wf = WorkflowCompiler(name="loop-root-merge", entry="head")

    @wf.task
    async def head(ctx) -> dict:
        seen_inputs.append(dict(ctx.inputs))
        n = ctx.inputs.get("n", 0) + 1
        return {"n": n}

    @wf.task(depends_on=["head"])
    async def check(ctx) -> tuple[dict, Next]:
        n = ctx.inputs["n"]
        return {"n": n}, Next("exit" if n >= 2 else "continue")

    wf.loop(body=["head"], until="check", max_iters=5)

    result = await WorkflowRuntime().execute(
        wf.compile(), run_context=_RunContextStub(tmp_path / "run")
    )
    assert result.status == "completed"
    assert len(seen_inputs) == 2
    # First iteration: engine-injected root inputs only.
    assert seen_inputs[0]["params"] == {"alpha": 7}
    assert "n" not in seen_inputs[0]
    # Second iteration: params preserved, loop-delivered key merged in.
    assert seen_inputs[1]["params"] == {"alpha": 7}
    assert seen_inputs[1]["n"] == 1
