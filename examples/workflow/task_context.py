"""A tour of ``TaskContext`` — inputs, deps, config, workspace helpers.

Matches ``docs/guide/task-context.md``.

Shows how the same context object delivers all of:

* ``ctx.inputs``  — typed output from the upstream task
* ``ctx.deps``    — runtime-injected dependencies (passed via ``deps=`` kwarg)
* ``ctx.config``  — active ``ProfileConfig`` (read-only mapping)
* ``ctx.artifact`` / ``ctx.log`` / ``ctx.set_result`` — workspace helpers

Run directly::

    python examples/workflow/task_context.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import molexp as me
from molexp.config import ProfileConfig
from molexp.workflow import Task, TaskContext, Workflow


@dataclass
class Deps:
    """Injected dependencies — anything you'd otherwise thread manually."""

    prefix: str


class Seed(Task):
    """Root task: produces the initial value."""

    async def execute(self, ctx: TaskContext[None, Deps, None]) -> int:
        return 1


class Record(Task):
    """Downstream task: reads ``ctx.inputs``, ``ctx.deps``, ``ctx.config``."""

    async def execute(self, ctx: TaskContext[None, Deps, int]) -> int:
        label = f"{ctx.deps.prefix}-{ctx.inputs}"
        scale = ctx.config.get("scale", 1)
        value = ctx.inputs * scale

        # Workspace helpers — no-ops if no Run is attached.
        ctx.artifact.save(f"{label}.json", {"value": value})
        ctx.log("record").append(label)
        ctx.set_result(label, value)
        return value


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-ctx-"))
    ws = me.Workspace(root, name="ctx-demo")
    project = ws.project("demo")
    exp = project.experiment("counter")

    spec = Workflow(name="counter").add(Seed()).add(Record(), depends_on=["seed"]).build()
    exp.set_workflow(spec)

    run = exp.run()
    result = await spec.execute(
        run=run,
        profile_config=ProfileConfig({"scale": 10}, name="smoke"),
        deps=Deps(prefix="step"),
    )

    run_json = json.loads((run.run_dir / "run.json").read_text())
    print(f"status:  {result.status}")
    print(f"outputs: {result.outputs}")
    print(f"results: {run_json['context']['results']}")


if __name__ == "__main__":
    asyncio.run(main())
