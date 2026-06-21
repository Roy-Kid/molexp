"""What actually appears on disk when a ``Run`` is tracked.

Matches ``docs/getting-started/tracked-runs.md``.

Run directly::

    python examples/getting_started/03_tracked_run.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="baseline")


@wf.task
async def experiment_body(ctx: TaskContext) -> dict:
    seed = ctx.inputs["params"].get("seed", 0)
    return {"score": 0.87, "seed": seed}


compiled = wf.compile()


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-tracked-"))
    print(f"workspace root: {root}\n")

    ws = me.Workspace(root, name="tracked-demo")
    exp = ws.project("demo").experiment("baseline").run(compiled, params={"seed": [42]})

    run = exp.list_runs()[0]
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)
        # Driver-side workspace helpers — results, artifacts, logs.
        ctx.set_result("score", result.outputs["experiment_body"]["score"])
        ctx.artifact.save("report.txt", "summary goes here")
        ctx.log("train").append("epoch 1 complete")

    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(path.relative_to(root))

    print("\nselected run fields (public API)")
    print(f"  id:              {run.id}")
    print(f"  status:          {run.status}")
    print(f"  parameters:      {run.parameters}")
    print(f"  profile:         {run.metadata.profile}")
    print(f"  execution count: {len(run.execution_history)}")
    print(f"  score:           {run.get_result('score')}")


if __name__ == "__main__":
    asyncio.run(main())
