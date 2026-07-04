"""Public-API contract — ``wf.branch`` / ``wf.loop`` / ``Next`` are blessed.

``Next`` graduated from IR-internal token to public routing return value:
it is importable from ``molexp.workflow`` (and listed in ``__all__``), and a
branch or loop workflow is buildable end-to-end from public imports alone.

Pinned semantics:

* a branch task returns ``(value, Next(label))``; the routed target receives
  ``value`` as ``ctx.inputs`` (values-on-edges delivery);
* ``wf.loop(body=..., until=..., max_iters=...)``: the ``until`` task returns
  ``Next("continue")`` to re-run the body or ``Next("exit")`` to proceed to
  ``on_exit``; loop-back values arrive at the body head via ``ctx.inputs``;
* ``max_iters`` forces exit with a :class:`LoopMaxItersExceeded` warning
  rather than failing the workflow.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    Next,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)

# ── public surface ───────────────────────────────────────────────────────────


# ── branch: public imports, routed value via ctx.inputs ─────────────────────


@pytest.mark.asyncio
async def test_branch_workflow_from_public_imports() -> None:
    """A branch workflow built only from public imports executes end-to-end;
    the ``(value, Next(label))`` payload reaches the routed target's
    ``ctx.inputs``."""
    seen: dict[str, object] = {}

    wf = WorkflowCompiler(name="public-branch", entry="classify")

    @wf.task
    async def classify(ctx: TaskContext) -> tuple[dict, Next]:
        return {"score": 0.9}, Next("accept")

    @wf.task
    async def accepted(**inputs: object) -> dict:
        seen["inputs"] = inputs
        return inputs

    @wf.task
    async def rejected(ctx: TaskContext) -> None:
        seen["rejected_ran"] = True

    wf.branch("classify", routes={"accept": "accepted", "reject": "rejected"})

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "succeeded"
    assert seen["inputs"] == {"score": 0.9}, (
        "the branch-routed value must arrive at the target as ctx.inputs"
    )
    assert "rejected_ran" not in seen, "the un-routed branch target must not run"


@pytest.mark.asyncio
async def test_branch_single_edge_form() -> None:
    """``wf.branch(src, label, to)`` (positional single-edge form) also works."""
    wf = WorkflowCompiler(name="public-branch-single", entry="src")

    @wf.task
    async def src(ctx: TaskContext) -> tuple[str, Next]:
        return "payload", Next("go")

    @wf.task
    async def dst(value: str) -> str:
        return f"got:{value}"

    wf.branch("src", "go", "dst")

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "succeeded"
    assert result.outputs["dst"] == "got:payload"


# ── loop: until-condition semantics + loop-back via ctx.inputs ──────────────


@pytest.mark.asyncio
async def test_loop_workflow_from_public_imports() -> None:
    """A ``wf.loop`` built only from public imports: the until-task returns
    ``Next("continue")`` until the condition holds, then ``Next("exit")``
    routes to ``on_exit``; each iteration's value reaches the next iteration's
    body head via ``ctx.inputs``."""
    head_inputs: list[object] = []

    wf = WorkflowCompiler(name="public-loop", entry="step")

    @wf.task
    async def step(value: int | None = None) -> int:
        head_inputs.append(value)
        prev = value if isinstance(value, int) else 0
        return prev + 1

    @wf.task(depends_on=["step"])
    async def check(step: int) -> tuple[int, Next]:
        n = step
        return n, Next("exit" if n >= 3 else "continue")

    @wf.task
    async def report(value: int) -> str:
        return f"final:{value}"

    wf.loop(body=["step"], until="check", max_iters=10, on_exit="report")

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "succeeded"
    # First iteration has no incoming value; later ones see the previous
    # iteration's routed output as ctx.inputs (values-on-edges).
    assert head_inputs == [None, 1, 2]
    assert result.outputs["step"] == 3
    assert result.outputs["report"] == "final:3"
