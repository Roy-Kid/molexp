"""What actually appears on disk when a ``Run`` is tracked.

Matches ``docs/getting-started/tracked-runs.md``.

Run directly::

    python examples/getting_started/03_tracked_run.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import promote_callable


async def experiment_body(ctx: me.RunContext) -> None:
    ctx.set_result("score", 0.87)
    ctx.artifact.save("report.txt", "summary goes here")
    ctx.log("train").append("epoch 1 complete")


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-tracked-"))
    print(f"workspace root: {root}\n")

    ws = me.Workspace(root, name="tracked-demo")
    project = ws.add_project("demo")
    exp = project.add_experiment("baseline", params={"lr": 1e-3})
    spec = promote_callable(experiment_body, name="experiment_body")
    spec.bind_to(exp)

    run = exp.add_run(parameters={"seed": 42})
    with run.start() as ctx:
        await spec.execute(run_context=ctx)

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
