"""Dependency-barrier deadlock guard (workflow-workspace-hardening P0-3 / ac-002).

The per-Step dependency barrier (``compiler._make_step_fn``) waits for every
declared ``depends_on`` to record a result before running a coalescing-Join
Step. A branch that skips an upstream means that upstream *never* records — the
barrier used to ``asyncio.sleep``-poll forever, hanging the whole run.

The guard detects *frontier exhaustion*: when no task body is executing
(``state.running == 0``) and no result has been recorded for a sustained
window while a dependency is still missing, nothing will ever satisfy it, so
the run raises :class:`WorkflowDeadlockError` (naming the unsatisfied deps)
within bounded time instead of hanging. A genuinely slow upstream keeps
``running > 0`` and never trips the guard.
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
        # wait_for bounds the RED case: pre-fix this hangs, so a TimeoutError
        # (not WorkflowDeadlockError) surfaces and the test fails. Post-fix the
        # guard raises in ~0.5s, well under the ceiling.
        await asyncio.wait_for(WorkflowRuntime().execute(wf.compile()), timeout=10)

    # The error must name the unsatisfied dependency.
    assert "bad" in str(excinfo.value)


@pytest.mark.asyncio
async def test_slow_upstream_does_not_trip_guard():
    """A genuinely slow upstream keeps ``running > 0`` and must NOT be mistaken
    for an absent dependency — the guard is frontier-exhaustion, not a timeout.

    ``slow`` sleeps well past the quiescence window (~0.5s); a naive timeout
    barrier would falsely raise. The join must complete normally instead.
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
