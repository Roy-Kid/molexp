"""What survives re-opening a run: ``run.json`` + ``execution_history``.

Matches ``docs/guide/workflow-persistence.md``.

Executes the same run twice (first failing, then succeeding) and prints
the fields in ``run.json`` that let you trace the attempt history,
profile metadata, and deterministic config hash.

Run directly::

    python examples/workspace/workflow_persistence.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import molexp as me
from molexp.config import ProfileConfig


def flaky_train(fail_once_path: Path):
    """Build a task that fails the first attempt, succeeds the second."""

    async def run_body(ctx: me.RunContext) -> None:
        if not fail_once_path.exists():
            fail_once_path.touch()
            raise RuntimeError("first attempt boom")
        ctx.set_result("epochs", ctx.config["epochs"])

    return run_body


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-persist-"))
    marker = root / "fail-once"

    ws = me.Workspace(root, name="persist-demo")
    project = ws.project("demo")
    exp = project.experiment("train")
    exp.set_workflow(flaky_train(marker))

    cfg = ProfileConfig({"epochs": 5}, name="smoke")
    run = exp.run(parameters={"seed": 0})

    # Attempt 1 — fails.
    result = await exp.workflow.execute(run=run, profile_config=cfg)
    print(f"attempt 1: status={result.status}")

    # Attempt 2 — same run, re-opened via ``Run.open`` semantics.
    result = await exp.workflow.execute(run=run, profile_config=cfg)
    print(f"attempt 2: status={result.status}")

    run_json = json.loads((run.run_dir / "run.json").read_text())
    print("\nrun.json (selected fields)")
    print(f"  id:           {run_json['id']}")
    print(f"  status:       {run_json['status']}")
    print(f"  profile:      {run_json['profile']}")
    print(f"  config_hash:  {run_json['config_hash']}")
    print(f"  config:       {run_json['config']}")
    print(f"  attempts:     {len(run_json['execution_history'])}")
    for i, entry in enumerate(run_json["execution_history"]):
        print(
            f"    #{i + 1}: status={entry['status']}, "
            f"started={entry['started_at']}, finished={entry.get('finished_at')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
