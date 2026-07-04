"""Workflow failure preserves completed-task results (P1-6 / ac-011).

When a task body raises, the exception propagates out of the graph runner.
``WorkflowRuntime.execute`` used to discard everything and return
``outputs={}`` — throwing away the (often expensive) results of every task
that already finished. The in-place-mutated ``WorkflowState`` still holds
those results, so the failed ``WorkflowResult`` now carries them, letting the
caller resume via ``seed_outputs=`` instead of recomputing from scratch.
"""

from __future__ import annotations

import pytest

from molexp.workflow import WorkflowCompiler, WorkflowRuntime


@pytest.mark.asyncio
async def test_failure_preserves_completed_results():
    """A raising downstream task leaves the completed upstream's output in the
    failed result's ``outputs`` (not an empty dict)."""
    wf = WorkflowCompiler(name="partial")

    @wf.task
    async def good(ctx) -> str:
        return "good-out"

    @wf.task(depends_on=["good"])
    async def boom(ctx) -> str:
        raise RuntimeError("kaboom")

    result = await WorkflowRuntime().execute(wf.compile())

    assert result.status == "failed"
    assert result.outputs.get("good") == "good-out"  # preserved, not dropped
