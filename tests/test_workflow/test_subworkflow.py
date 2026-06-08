"""Tests for SubWorkflow — the sanctioned sub-workflow composition node (spec 06).

Contract:

* ``SubWorkflow`` is a real exported node (subclass of ``Task``).
* ``SubWorkflow(inner).execute(ctx)`` runs the inner spec end-to-end through the
  engine and returns the inner terminal output (default: the single dependency
  leaf; or ``output=`` when set).
* It forwards the outer ``run_context`` by identity — no hand-built
  ``TaskContext`` clone (the source contains no ``TaskContext(`` call).
* It slots into ``builder.parallel(body="sub", ...)`` as the per-element body
  with no per-element node growth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workflow import (
    SubWorkflow,
    Task,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)

# ── ac-001: export + subclass ────────────────────────────────────────────────


def test_subworkflow_exported_and_is_task_subclass() -> None:
    """ac-001 — SubWorkflow imports from molexp.workflow and subclasses Task."""
    assert issubclass(SubWorkflow, Task)


def test_subworkflow_source_has_no_taskcontext_construction() -> None:
    """ac-003 (negative) — subworkflow.py never constructs a TaskContext."""
    source = (
        Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow" / "subworkflow.py"
    ).read_text()
    assert "TaskContext(" not in source


# ── Helpers: inner workflows ─────────────────────────────────────────────────


def _build_multi_step_inner() -> WorkflowCompiler:
    """A >=2-task inner chain (TypifyMonomer-equivalent): load → normalize → scale."""
    wf = WorkflowCompiler(name="inner-multi")

    @wf.task
    async def load(ctx: TaskContext) -> list[float]:
        return [2.0, 4.0, 8.0]

    @wf.task(depends_on=["load"])
    async def normalize(ctx: TaskContext) -> list[float]:
        top = max(ctx.inputs)
        return [x / top for x in ctx.inputs]

    @wf.task(depends_on=["normalize"])
    async def scale(ctx: TaskContext) -> float:
        return sum(ctx.inputs)

    return wf


def _build_single_step_inner() -> WorkflowCompiler:
    """A 1-task inner spec (Embed3D-equivalent)."""
    wf = WorkflowCompiler(name="inner-single")

    @wf.task
    async def embed(ctx: TaskContext) -> int:
        return 42

    return wf


# ── ac-002 / ac-005: multi-step inner returns terminal output ────────────────


@pytest.mark.asyncio
async def test_subworkflow_runs_multi_step_inner_returns_terminal() -> None:
    """ac-002 / ac-005 — multi-task inner chain returns its terminal output."""
    outer = (
        WorkflowCompiler(name="outer-multi")
        .add(SubWorkflow(_build_multi_step_inner()), name="sub")
        .compile()
    )
    result = await WorkflowRuntime().execute(outer)
    assert result.status == "completed"
    # load=[2,4,8] → normalize=[0.25,0.5,1.0] → scale=1.75
    assert result.outputs["sub"] == pytest.approx(1.75)


@pytest.mark.asyncio
async def test_subworkflow_single_step_inner_returns_terminal() -> None:
    """ac-005 — a 1-task inner spec behaves identically via the same class."""
    outer = (
        WorkflowCompiler(name="outer-single")
        .add(SubWorkflow(_build_single_step_inner()), name="sub")
        .compile()
    )
    result = await WorkflowRuntime().execute(outer)
    assert result.status == "completed"
    assert result.outputs["sub"] == 42


@pytest.mark.asyncio
async def test_subworkflow_accepts_compiled_workflow() -> None:
    """ac-002 — SubWorkflow accepts an already-compiled inner workflow too."""
    inner_compiled = _build_multi_step_inner().compile()
    outer = (
        WorkflowCompiler(name="outer-compiled")
        .add(SubWorkflow(inner_compiled), name="sub")
        .compile()
    )
    result = await WorkflowRuntime().execute(outer)
    assert result.status == "completed"
    assert result.outputs["sub"] == pytest.approx(1.75)


@pytest.mark.asyncio
async def test_subworkflow_explicit_output_selection() -> None:
    """ac-002 — output= selects a specific inner task's output."""
    outer = (
        WorkflowCompiler(name="outer-explicit")
        .add(SubWorkflow(_build_multi_step_inner(), output="normalize"), name="sub")
        .compile()
    )
    result = await WorkflowRuntime().execute(outer)
    assert result.status == "completed"
    assert result.outputs["sub"] == pytest.approx([0.25, 0.5, 1.0])


@pytest.mark.asyncio
async def test_subworkflow_feeds_downstream_task() -> None:
    """ac-002 — add(SubWorkflow) → add(downstream) makes the inner result flow."""

    class Double(Task):
        async def execute(self, ctx: TaskContext) -> float:
            return ctx.inputs * 2

    outer = (
        WorkflowCompiler(name="outer-downstream")
        .add(SubWorkflow(_build_multi_step_inner()), name="sub")
        .add(Double(), name="double", depends_on=["sub"])
        .compile()
    )
    result = await WorkflowRuntime().execute(outer)
    assert result.status == "completed"
    assert result.outputs["double"] == pytest.approx(3.5)


def test_subworkflow_ambiguous_leaf_raises() -> None:
    """An inner spec with multiple leaves and no output= raises a clear error."""
    inner = WorkflowCompiler(name="inner-two-leaves")

    @inner.task
    async def seed(ctx: TaskContext) -> int:
        return 1

    @inner.task(depends_on=["seed"])
    async def leaf_a(ctx: TaskContext) -> int:
        return ctx.inputs + 1

    @inner.task(depends_on=["seed"])
    async def leaf_b(ctx: TaskContext) -> int:
        return ctx.inputs + 2

    node = SubWorkflow(inner)
    with pytest.raises(ValueError, match="leaf"):
        node._resolve_output_name()


# ── ac-003: run_context forwarded by identity ────────────────────────────────


@pytest.mark.asyncio
async def test_subworkflow_forwards_run_context_by_identity() -> None:
    """ac-003 — inner task observes the SAME run_context object the outer received."""
    captured: list[object] = []

    inner = WorkflowCompiler(name="inner-rc")

    @inner.task
    async def observe(ctx: TaskContext) -> str:
        captured.append(ctx.run_context)
        return "ok"

    outer = WorkflowCompiler(name="outer-rc").add(SubWorkflow(inner), name="sub").compile()

    sentinel = object()
    result = await WorkflowRuntime().execute(outer, run_context=sentinel)
    assert result.status == "completed"
    assert len(captured) == 1
    assert captured[0] is sentinel


# ── ac-004: SubWorkflow as a parallel body ───────────────────────────────────


@pytest.mark.asyncio
async def test_subworkflow_as_parallel_body() -> None:
    """ac-004 — SubWorkflow is the per-element body of builder.parallel.

    Every map_over element drives the full inner chain once; ``join`` receives
    one inner output per element in iteration order; the compiled task set is
    exactly the declared outer tasks (no per-element node growth).
    """
    wf = WorkflowCompiler(name="outer-parallel", entry="enumerate")

    @wf.task
    async def enumerate(ctx: TaskContext) -> list[int]:
        return [0, 1, 2, 3]  # 4 elements

    # The body is a SubWorkflow over a >=2-task inner chain. It runs the full
    # fixed inner chain per element and returns its terminal (1.75).
    wf.add(SubWorkflow(_build_multi_step_inner()), name="sub")

    @wf.task
    async def collect(ctx: TaskContext) -> list[float]:
        return list(ctx.inputs)

    wf.parallel(map_over="enumerate", body="sub", join="collect", max_concurrency=2)

    compiled = wf.compile()

    # No per-element node growth — exactly the declared outer task set.
    assert {t.name for t in compiled._tasks} == {"enumerate", "sub", "collect"}

    result = await WorkflowRuntime().execute(compiled)
    assert result.status == "completed"
    # One inner output (1.75) per element, in iteration order.
    assert result.outputs["sub"] == pytest.approx([1.75, 1.75, 1.75, 1.75])
    assert result.outputs["collect"] == pytest.approx([1.75, 1.75, 1.75, 1.75])


@pytest.mark.asyncio
async def test_subworkflow_parallel_body_single_task_inner() -> None:
    """ac-004 / ac-005 — a 1-task inner spec also works as a parallel body."""
    wf = WorkflowCompiler(name="outer-parallel-single", entry="enumerate")

    @wf.task
    async def enumerate(ctx: TaskContext) -> list[int]:
        return [0, 1, 2]

    wf.add(SubWorkflow(_build_single_step_inner()), name="sub")

    @wf.task
    async def collect(ctx: TaskContext) -> list[int]:
        return list(ctx.inputs)

    wf.parallel(map_over="enumerate", body="sub", join="collect", max_concurrency=3)

    compiled = wf.compile()
    assert {t.name for t in compiled._tasks} == {"enumerate", "sub", "collect"}

    result = await WorkflowRuntime().execute(compiled)
    assert result.status == "completed"
    assert result.outputs["collect"] == [42, 42, 42]


@pytest.mark.asyncio
async def test_subworkflow_parallel_element_failure_surfaces() -> None:
    """Edge — a per-element inner failure surfaces via ParallelExecutionError."""
    from molexp.workflow import ParallelExecutionError

    sibling_runs: list[int] = []

    # Inner spec that fails based on a shared mutable counter so exactly one
    # element raises while siblings complete.
    state = {"n": 0}

    inner = WorkflowCompiler(name="inner-maybe-fail")

    @inner.task
    async def step(ctx: TaskContext) -> int:
        idx = state["n"]
        state["n"] += 1
        if idx == 1:
            raise ValueError("boom")
        sibling_runs.append(idx)
        return idx

    wf = WorkflowCompiler(name="outer-parallel-fail", entry="enumerate")

    @wf.task
    async def enumerate(ctx: TaskContext) -> list[int]:
        return [0, 1, 2]

    wf.add(SubWorkflow(inner), name="sub")

    @wf.task
    async def collect(ctx: TaskContext) -> int:
        return len(ctx.inputs)

    wf.parallel(map_over="enumerate", body="sub", join="collect", max_concurrency=1)

    with pytest.raises(ParallelExecutionError) as exc_info:
        await WorkflowRuntime().execute(wf.compile())

    assert exc_info.value.body == "sub"
    assert set(exc_info.value.failures.keys()) == {1}
