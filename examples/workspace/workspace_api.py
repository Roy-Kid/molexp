"""The ``Workspace → Project → Experiment → Run`` walk, end to end.

Matches ``docs/guide/workspace-api.md``.

Shows the idempotent factory methods (``ws.project``, ``project.experiment``,
``exp.run``) and how re-opening a workspace returns the same logical
entities instead of creating duplicates.

Run directly::

    python examples/workspace/workspace_api.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me


async def step(ctx: me.RunContext) -> None:
    ctx.set_result("noted", True)


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-ws-api-"))
    print(f"workspace root: {root}\n")

    ws = me.Workspace(root, name="ws-api-demo")
    project = ws.project("demo")
    exp = project.experiment("baseline", params={"lr": 1e-3})
    exp.set_workflow(step)
    r1 = exp.run(parameters={"seed": 0})
    r2 = exp.run(parameters={"seed": 1})

    for run in (r1, r2):
        with run.start() as ctx:
            await exp.workflow.execute(run_context=ctx)

    # Idempotent factories: calling them again returns the same entity.
    same_project = ws.project("demo")
    same_exp = same_project.experiment("baseline", params={"lr": 1e-3})
    assert same_project is project
    assert same_exp is exp
    print("ws.project('demo') is idempotent:", same_project is project)

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
