"""A tour of ``TaskContext`` — the pure ``{inputs, config}`` contract.

Matches ``docs/guide/task-context.md``.

Every task body receives one frozen ``TaskContext``:

* ``ctx.inputs``  — runtime data flowing in along the graph's edges. For a
  downstream task that is the upstream output; for a root task of a tracked
  run the engine injects ``{"params": <run params>, "workdir": <Path>}``.
* ``ctx.config``  — the active configuration mapping (the resolved profile
  for a tracked run, or the ``config=`` kwarg of ``WorkflowRuntime.execute``).
* ``ctx.workdir`` — content-addressed scratch directory for this task
  (``None`` when no workspace run is attached).

There is no ``ctx.run_context`` and no ``ctx.deps``: a task cannot climb up to
the Run or the workspace. Workspace helpers (``set_result`` / ``artifact`` /
``log``) live on the driver-side ``RunContext`` instead.

Run directly::

    python examples/workflow/task_context.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.profile import ProfileConfig
from molexp.workflow import Task, TaskContext, WorkflowCompiler, WorkflowRuntime


class Seed(Task):
    """Root task: reads the engine-injected ``{"params", "workdir"}`` inputs."""

    async def execute(self, ctx: TaskContext) -> int:
        base = ctx.inputs["params"].get("base", 1)
        workdir = ctx.inputs["workdir"]  # same Path as ctx.workdir
        if workdir is not None:
            (workdir / "seed.txt").write_text(str(base))
        return base


class Record(Task):
    """Downstream task: reads ``ctx.inputs`` (upstream output) and ``ctx.config``."""

    async def execute(self, ctx: TaskContext) -> int:
        scale = ctx.config.get("scale", 1)
        return ctx.inputs * scale


# Module scope so the compiled artifact is importable across CLI re-imports.
compiled = WorkflowCompiler(name="counter").add(Seed()).add(Record(), depends_on=["seed"]).compile()


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-ctx-"))
    ws = me.Workspace(root, name="ctx-demo")
    exp = ws.project("demo").experiment("counter").run(compiled, params={"base": [1]})

    run = exp.list_runs()[0]
    cfg = ProfileConfig({"scale": 10}, name="smoke")
    with run.start(profile_config=cfg) as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)
        # Workspace helpers are driver-side, on the RunContext.
        ctx.set_result("record", result.outputs["record"])
        ctx.artifact.save("record.json", {"value": result.outputs["record"]})
        ctx.log("record").append(f"value={result.outputs['record']}")

    print(f"status:  {result.status}")
    print(f"outputs: {result.outputs}")
    print(f"result:  {run.get_result('record')}")


if __name__ == "__main__":
    asyncio.run(main())
