"""What actually appears on disk when a ``Run`` is tracked.

Matches ``docs/getting-started/tracked-runs.md``.

Runs a tiny workflow under a materialised workspace, then prints the
relevant on-disk paths and the persisted ``run.json`` fields.

Run directly::

    python examples/getting_started/03_tracked_run.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import molexp as me


async def experiment_body(ctx: me.RunContext) -> None:
    ctx.set_result("score", 0.87)
    ctx.artifact.save("report.txt", "summary goes here")
    ctx.log("train").append("epoch 1 complete")


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-tracked-"))
    print(f"workspace root: {root}\n")

    ws = me.Workspace(root, name="tracked-demo")
    project = ws.project("demo")
    exp = project.experiment("baseline", params={"lr": 1e-3})
    exp.set_workflow(experiment_body)

    run = exp.run(parameters={"seed": 42})
    await exp.workflow.execute(run=run)

    # Layout that ``molexp run`` would produce.
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(path.relative_to(root))

    run_json = json.loads((run.run_dir / "run.json").read_text())
    print("\nselected run.json fields")
    print(f"  id:              {run_json['id']}")
    print(f"  status:          {run_json['status']}")
    print(f"  parameters:      {run_json['parameters']}")
    print(f"  profile:         {run_json['profile']}")
    print(f"  execution count: {len(run_json['execution_history'])}")
    print(f"  results:         {run_json['context']['results']}")


if __name__ == "__main__":
    asyncio.run(main())
