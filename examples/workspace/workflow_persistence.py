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

# Module-level marker so the body is importable as a fresh callable on
# every attempt — ``Experiment.set_workflow`` captures an entrypoint.
_FAIL_ONCE_MARKER: Path | None = None


async def flaky_train(ctx: me.RunContext) -> None:
    assert _FAIL_ONCE_MARKER is not None
    if not _FAIL_ONCE_MARKER.exists():
        _FAIL_ONCE_MARKER.touch()
        raise RuntimeError("first attempt boom")
    ctx.set_result("epochs", ctx.config["epochs"])


async def main() -> None:
    global _FAIL_ONCE_MARKER

    root = Path(tempfile.mkdtemp(prefix="molexp-persist-"))
    _FAIL_ONCE_MARKER = root / "fail-once"

    ws = me.Workspace(root, name="persist-demo")
    project = ws.project("demo")
    exp = project.experiment("train")
    exp.set_workflow(flaky_train)

    cfg = ProfileConfig({"epochs": 5}, name="smoke")
    run = exp.run(parameters={"seed": 0})

    # ``execute()`` captures task failures and records them on the run
    # without re-raising — inspect ``result.status`` instead.
    with run.start(profile_config=cfg) as ctx:
        result = await exp.workflow.execute(run_context=ctx)
    print(f"attempt 1: status={result.status}")

    with run.start(profile_config=cfg) as ctx:
        result = await exp.workflow.execute(run_context=ctx)
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
