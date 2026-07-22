"""Structural deadlock detection (values-on-edges engine).

The engine launches a task only when its trigger edges have fired AND its
declared ``depends_on`` values are present. A branch that routes away from an
upstream makes that upstream *structurally dead* — from the graph alone the
engine knows no live path can ever record its output, so a consumer blocking
on it raises :class:`WorkflowDeadlockError` (naming the unsatisfied deps) the
moment it becomes control-ready: deterministic, zero timing constants. The
dual guarantee is that a genuinely slow-but-live upstream is never mistaken
for an absent one.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.workflow import WorkflowCompiler, WorkflowRuntime
from molexp.workflow.types import Next, WorkflowDeadlockError


class TestStructuralDeadlockDetection:
    @pytest.mark.asyncio
    async def test_branch_skipped_dep_raises_rather_than_hangs(self) -> None:
        """A join depending on a branch-skipped task raises
        ``WorkflowDeadlockError`` (naming the dep) in bounded time rather than
        busy-polling forever."""
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
            # wait_for bounds the regression: if detection ever stopped being
            # structural this would hang and surface as TimeoutError instead.
            await asyncio.wait_for(WorkflowRuntime().execute(wf.compile()), timeout=10)

        assert "bad" in str(excinfo.value), "the error must name the unsatisfied dependency"

    @pytest.mark.asyncio
    async def test_slow_upstream_is_not_mistaken_for_absent_dep(self) -> None:
        """A genuinely slow but live upstream must NOT trip the guard —
        detection is structural, never a timeout. ``slow`` sleeps past any
        quiescence window; the join must still complete normally."""
        wf = WorkflowCompiler(name="slow-ok")

        @wf.task
        async def fast(ctx) -> str:
            return "fast-out"

        @wf.task
        async def slow(ctx) -> str:
            await asyncio.sleep(1.0)  # body stays in flight
            return "slow-out"

        @wf.task(depends_on=["fast", "slow"])
        async def join(ctx) -> str:
            return "joined"

        result = await asyncio.wait_for(WorkflowRuntime().execute(wf.compile()), timeout=10)
        assert result.status == "succeeded"
        assert result.outputs["join"] == "joined"
        assert result.outputs["slow"] == "slow-out"
