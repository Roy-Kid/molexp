"""Structural deadlock detection (values-on-edges engine).

The engine launches a task only when its trigger edges have fired AND its
declared ``depends_on`` values are present. A branch that routes away from an
upstream makes that upstream *structurally dead* — the engine knows, from the
graph alone, that no live path can ever record its output. A consumer
blocking on a dead dependency raises :class:`WorkflowDeadlockError` (naming
the unsatisfied deps) the moment it becomes control-ready — deterministic,
with zero timing constants. A genuinely slow upstream is simply a live node
still running; structural detection can never mistake it for an absent one.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.workflow import WorkflowCompiler, WorkflowRuntime
from molexp.workflow.types import Next, WorkflowDeadlockError


@pytest.mark.asyncio
async def test_branch_skipped_dep_raises_deadlock_not_hang():
    """A join depending on a branch-skipped task raises WorkflowDeadlockError
    in bounded time rather than busy-polling forever."""
    wf = WorkflowCompiler(name="deadlock", entry="route")

    @wf.task(routes={"ok": "good", "fail": "bad"})
    async def route(ctx) -> Next:
        return Next("ok")  # `bad` never runs → never records

    @wf.task
    async def good(ctx) -> str:
        return "good-ran"

    @wf.task
    async def bad(ctx) -> str:
        return "bad-ran"

    @wf.task(depends_on=["good", "bad"])
    async def join(ctx) -> str:
        return "joined"

    with pytest.raises(WorkflowDeadlockError) as excinfo:
        # wait_for bounds the regression case: if detection ever stopped being
        # structural this would hang and surface as TimeoutError instead.
        # Structural detection raises immediately (no quiescence window).
        await asyncio.wait_for(WorkflowRuntime().execute(wf.compile()), timeout=10)

    # The error must name the unsatisfied dependency.
    assert "bad" in str(excinfo.value)


@pytest.mark.asyncio
async def test_slow_upstream_does_not_trip_guard():
    """A genuinely slow upstream is a live node and must NOT be mistaken for
    an absent dependency — detection is structural, never a timeout.

    ``slow`` sleeps 1s; any timing-based guard would be tempted to misfire.
    The join must complete normally instead.
    """
    wf = WorkflowCompiler(name="slow-ok")

    @wf.task
    async def fast(ctx) -> str:
        return "fast-out"

    @wf.task
    async def slow(ctx) -> str:
        await asyncio.sleep(1.0)  # > quiescent window; body stays in flight
        return "slow-out"

    @wf.task(depends_on=["fast", "slow"])
    async def join(ctx) -> str:
        return "joined"

    result = await asyncio.wait_for(WorkflowRuntime().execute(wf.compile()), timeout=10)
    assert result.status == "completed"
    assert result.outputs["join"] == "joined"
    assert result.outputs["slow"] == "slow-out"
