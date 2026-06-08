"""Integration test for the resume roundtrip (verb C).

Composes the three new pieces end to end:

1. First attempt fails after ``good`` completes → ``workflow.json`` records it.
2. ``read_node_outputs`` recovers ``good``'s output from the failed attempt.
3. ``run.start(execution_id=exec1)`` REOPENS the same execution record.
4. ``WorkflowRuntime.execute(..., seed_outputs=seed, execution_id=exec1)``
   skips the seeded ``good`` body and completes the fixed ``boom``.

Tasks carry ``task_type`` slugs because persisting ``workflow.json`` requires
the compiled workflow to be IR-serializable — that file is precisely what
``read_node_outputs`` reads the seed back from.
"""

from __future__ import annotations

import pytest

from molexp.workflow import Task, TaskContext, WorkflowCompiler, WorkflowRuntime
from molexp.workflow._pydantic_graph.persistence import read_node_outputs
from molexp.workspace import Workspace


@pytest.mark.asyncio
async def test_resume_roundtrip_seeds_and_reopens(tmp_path) -> None:
    ws = Workspace(root=tmp_path, name="x")
    p = ws.add_project("p")
    e = p.add_experiment("e", workflow_source="train.py", params={}, git_commit="c")
    run = e.add_run(parameters={})

    good_runs = {"n": 0}

    class Good(Task):
        async def execute(self, ctx: TaskContext) -> str:
            good_runs["n"] += 1
            return "good-out"

    class Boom(Task):
        async def execute(self, ctx: TaskContext) -> str:
            raise RuntimeError("kaboom")

    class BoomFixed(Task):
        async def execute(self, ctx: TaskContext) -> str:
            return "boom-fixed"

    # ── Attempt 1: good completes, boom raises ──────────────────────────────
    wf1 = (
        WorkflowCompiler(name="attempt-1")
        .add(Good(), name="good", task_type="test.good")
        .add(Boom(), name="boom", task_type="test.boom", depends_on=["good"])
    )

    with run.start() as ctx:
        res1 = await WorkflowRuntime().execute(wf1.compile(), run_context=ctx)

    assert res1.status == "failed"
    assert good_runs["n"] == 1

    exec1 = run.metadata.execution_history[-1].execution_id

    # Read the seed BEFORE re-executing (execute() overwrites workflow.json).
    seed = read_node_outputs(run.run_dir, exec1)
    assert seed.get("good") == "good-out"

    # ── Attempt 2: reopen + reseed, boom now succeeds ───────────────────────
    wf2 = (
        WorkflowCompiler(name="attempt-2")
        .add(Good(), name="good", task_type="test.good")
        .add(BoomFixed(), name="boom", task_type="test.boom", depends_on=["good"])
    )

    with run.start(execution_id=exec1) as ctx:
        res2 = await WorkflowRuntime().execute(
            wf2.compile(),
            run_context=ctx,
            seed_outputs=seed,
            execution_id=exec1,
        )

    assert res2.status == "completed"
    assert res2.outputs["boom"] == "boom-fixed"
    assert good_runs["n"] == 1  # seeded on resume, body skipped
    assert len(run.metadata.execution_history) == 1  # reopened, not appended
