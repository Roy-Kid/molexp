"""A tour of ``TaskContext`` — inputs bind by name, ``ctx`` carries the workdir.

Matches ``docs/guide/task-context.md``.

A task body declares the runtime values it consumes as **named parameters**; the
engine binds them from the merged map {build-time config} | {upstream outputs |
run params} (dynamic inputs win):

* a root task's sweep params arrive as named params (``base`` below);
* a downstream task's single upstream output binds positionally to its sole free
  parameter (``value`` below);
* build-time / profile ``config`` fields bind by name (``scale`` below).

The only data surface left on the ``TaskContext`` itself is:

* ``ctx.workdir`` — a content-addressed scratch directory for this task
  (``None`` when no workspace run is attached). Keep the leading ``ctx``
  parameter only when the body writes there.

There is no ``ctx.inputs`` / ``ctx.config`` / ``ctx.run_context`` and no
``ctx.deps``: a task cannot climb up to the Run or the workspace. Workspace
helpers (``set_result`` / ``artifact`` / ``log``) live on the driver-side
``RunContext`` instead.

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
    """Root task: the run param ``base`` binds by name; ``ctx.workdir`` is used."""

    async def execute(self, ctx: TaskContext, base: int = 1) -> int:
        if ctx.workdir is not None:
            (ctx.workdir / "seed.txt").write_text(str(base))
        return base


class Record(Task):
    """Downstream task: ``value`` is the upstream output; ``scale`` is config."""

    async def execute(self, ctx: TaskContext, value: int, scale: int = 1) -> int:
        return value * scale


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
