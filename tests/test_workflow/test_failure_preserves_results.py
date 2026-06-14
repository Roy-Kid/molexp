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


@pytest.mark.asyncio
async def test_seed_outputs_resumes_from_preserved_results():
    """The preserved results feed ``seed_outputs`` so a resume skips the
    already-completed task and only runs the remainder."""
    good_runs = {"n": 0}

    wf1 = WorkflowCompiler(name="attempt-1")

    @wf1.task
    async def good(ctx) -> str:
        good_runs["n"] += 1
        return "good-out"

    @wf1.task(depends_on=["good"])
    async def boom(ctx) -> str:
        raise RuntimeError("kaboom")

    failed = await WorkflowRuntime().execute(wf1.compile())
    assert failed.status == "failed"
    assert failed.outputs.get("good") == "good-out"
    assert good_runs["n"] == 1

    # Resume: same shape, downstream now succeeds; seed the completed task.
    wf2 = WorkflowCompiler(name="attempt-2")

    @wf2.task
    async def good(ctx) -> str:  # noqa: F811 — same task name so seed matches
        good_runs["n"] += 1  # must NOT run again — it is seeded
        return "good-out"

    @wf2.task(depends_on=["good"])
    async def boom(ctx) -> str:  # noqa: F811 — same task name, now succeeds
        return "boom-fixed"

    resumed = await WorkflowRuntime().execute(wf2.compile(), seed_outputs=failed.outputs)

    assert resumed.status == "completed"
    assert resumed.outputs["boom"] == "boom-fixed"
    assert good_runs["n"] == 1  # seeded on resume, body skipped
