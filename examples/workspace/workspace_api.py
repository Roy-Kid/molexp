"""The ``Workspace → Project → Experiment → Run`` walk, end to end.

Matches ``docs/guide/workspace-api.md``.

Shows the idempotent fluent accessors (``ws.project``, ``project.experiment``),
the ``exp.run(workflow, params=...)`` sweep declaration, and how re-opening a
workspace returns the same logical entities instead of creating duplicates.

Run directly::

    python examples/workspace/workspace_api.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="step")


@wf.task
async def step(ctx: TaskContext) -> dict:
    return {"noted": True, "seed": ctx.inputs["params"].get("seed")}


compiled = wf.compile()


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-ws-api-"))
    print(f"workspace root: {root}\n")

    ws = me.Workspace(root, name="ws-api-demo")
    project = ws.project("demo")
    exp = project.experiment("baseline")

    # Declare the sweep: one content-addressed Run per parameter cell.
    exp.run(compiled, params={"seed": [0, 1]})
    runs = exp.list_runs()

    for run in runs:
        with run.start() as ctx:
            await WorkflowRuntime().execute(compiled, run_context=ctx)

    # Idempotent factories: calling them again returns the same entity.
    same_project = ws.project("demo")
    same_exp = same_project.experiment("baseline")
    print("ws.project('demo') is idempotent:        ", same_project is project)
    print("project.experiment('baseline') idempotent:", same_exp is exp)

    # Re-declaring the same sweep adds no duplicate runs.
    exp.run(compiled, params={"seed": [0, 1]})
    print("re-declaring the sweep adds no runs:      ", len(exp.list_runs()) == len(runs))

    # Reload the workspace from disk — same logical state.
    reopened = me.Workspace.load(root)
    print("\nafter Workspace.load(root):")
    print(f"  projects:    {[p.name for p in reopened.list_projects()]}")
    reopened_proj = reopened.get_project("demo")
    reopened_exp = reopened_proj.get_experiment(exp.id)
    print(f"  experiments: {[e.name for e in reopened_proj.list_experiments()]}")
    print(f"  runs:        {[r.id for r in reopened_exp.list_runs()]}")
    print(f"  run status:  {[r.status for r in reopened_exp.list_runs()]}")


if __name__ == "__main__":
    asyncio.run(main())
