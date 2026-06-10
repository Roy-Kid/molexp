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
    LoopMaxItersExceeded,
    Next,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)

# ── public surface ───────────────────────────────────────────────────────────


def test_next_is_public() -> None:
    """``Next`` is importable from ``molexp.workflow`` and listed in __all__."""
    import molexp.workflow as W

    assert "Next" in W.__all__, "Next is public API; it must be in molexp.workflow.__all__"
    from molexp.workflow.types import Next as InternalNext

    assert W.Next is InternalNext, "public Next must be the same class as workflow.types.Next"


def test_next_accepts_positional_label() -> None:
    """``Next("route")`` and ``Next(label="route")`` are equivalent spellings."""
    assert Next("go") == Next(label="go")
    assert Next("go").label == "go"


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
    async def accepted(ctx: TaskContext) -> dict:
        seen["inputs"] = ctx.inputs
        return ctx.inputs

    @wf.task
    async def rejected(ctx: TaskContext) -> None:
        seen["rejected_ran"] = True

    wf.branch("classify", routes={"accept": "accepted", "reject": "rejected"})

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
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
    async def dst(ctx: TaskContext) -> str:
        return f"got:{ctx.inputs}"

    wf.branch("src", "go", "dst")

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
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
    async def step(ctx: TaskContext) -> int:
        head_inputs.append(ctx.inputs)
        prev = ctx.inputs if isinstance(ctx.inputs, int) else 0
        return prev + 1

    @wf.task(depends_on=["step"])
    async def check(ctx: TaskContext) -> tuple[int, Next]:
        n = ctx.inputs
        return n, Next("exit" if n >= 3 else "continue")

    @wf.task
    async def report(ctx: TaskContext) -> str:
        return f"final:{ctx.inputs}"

    wf.loop(body=["step"], until="check", max_iters=10, on_exit="report")

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
    # First iteration has no incoming value; later ones see the previous
    # iteration's routed output as ctx.inputs (values-on-edges).
    assert head_inputs == [None, 1, 2]
    assert result.outputs["step"] == 3
    assert result.outputs["report"] == "final:3"


@pytest.mark.asyncio
async def test_loop_max_iters_forces_exit_with_warning() -> None:
    """``max_iters`` caps a runaway loop: the engine forces ``Next("exit")``
    and emits ``LoopMaxItersExceeded``; the workflow still completes."""
    runs = [0]

    wf = WorkflowCompiler(name="public-loop-runaway", entry="step")

    @wf.task
    async def step(ctx: TaskContext) -> int:
        runs[0] += 1
        return runs[0]

    @wf.task(depends_on=["step"])
    async def never_done(ctx: TaskContext) -> Next:
        return Next("continue")

    wf.loop(body=["step"], until="never_done", max_iters=3)

    with pytest.warns(LoopMaxItersExceeded):
        result = await WorkflowRuntime().execute(wf.compile())

    assert result.status == "completed"
    assert runs[0] == 3
