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
from molexp.workflow import promote_callable, Workflow


async def step(ctx: me.RunContext) -> None:
    ctx.set_result("noted", True)


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-ws-api-"))
    print(f"workspace root: {root}\n")

    ws = me.Workspace(root, name="ws-api-demo")
    project = ws.add_project("demo")
    exp = project.add_experiment("baseline", params={"lr": 1e-3})

    # Wrap the callable into a single-task Workflow, then bind the
    # spec to the experiment via the process-local registry.
    spec = promote_callable(step, name="step")
    spec.bind_to(exp)

    r1 = exp.add_run(parameters={"seed": 0})
    r2 = exp.add_run(parameters={"seed": 1})

    for run in (r1, r2):
        with run.start() as ctx:
            await spec.execute(run_context=ctx)

    # Idempotent factories: calling them again returns the same entity.
    same_project = ws.add_project("demo")
    same_exp = same_project.add_experiment("baseline", params={"lr": 1e-3})
    assert same_project is project
    assert same_exp is exp
    print("ws.add_project('demo') is idempotent:", same_project is project)

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
