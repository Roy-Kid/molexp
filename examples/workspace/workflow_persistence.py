"""What survives re-opening a run: ``run.json`` + ``execution_history``.

Matches ``docs/guide/workflow-persistence.md``.

Executes the same run twice (first failing, then succeeding) and prints
the public run fields that let you trace the attempt history, profile
metadata, and deterministic config hash.

Run directly::

    python examples/workspace/workflow_persistence.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.profile import ProfileConfig
from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

# Module-level marker so the first attempt fails and the second succeeds.
_FAIL_ONCE_MARKER: Path | None = None

wf = WorkflowCompiler(name="flaky")


@wf.task
async def flaky_train(ctx: TaskContext) -> dict:
    assert _FAIL_ONCE_MARKER is not None
    if not _FAIL_ONCE_MARKER.exists():
        _FAIL_ONCE_MARKER.touch()
        raise RuntimeError("first attempt boom")
    return {"epochs": ctx.config["epochs"]}


compiled = wf.compile()


async def main() -> None:
    global _FAIL_ONCE_MARKER

    root = Path(tempfile.mkdtemp(prefix="molexp-persist-"))
    _FAIL_ONCE_MARKER = root / "fail-once"

    ws = me.Workspace(root, name="persist-demo")
    exp = ws.project("demo").experiment("train").run(compiled, params={"seed": [0]})

    cfg = ProfileConfig({"epochs": 5}, name="smoke")
    run = exp.list_runs()[0]

    # ``execute()`` captures task failures and records them on the run
    # without re-raising — inspect ``result.status`` instead.
    with run.start(profile_config=cfg) as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)
    print(f"attempt 1: status={result.status}")

    with run.start(profile_config=cfg) as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)
    print(f"attempt 2: status={result.status}")

    print("\nrun fields (public API)")
    print(f"  id:           {run.id}")
    print(f"  status:       {run.status}")
    print(f"  profile:      {run.metadata.profile}")
    print(f"  config_hash:  {run.metadata.config_hash}")
    print(f"  config:       {run.metadata.config}")
    print(f"  attempts:     {len(run.metadata.execution_history)}")
    for i, entry in enumerate(run.metadata.execution_history):
        print(
            f"    #{i + 1}: status={entry.status}, "
            f"started={entry.started_at}, finished={entry.finished_at}"
        )


if __name__ == "__main__":
    asyncio.run(main())
