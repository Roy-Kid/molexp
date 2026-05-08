"""Inspect what files actually land on disk for one tracked run.

Matches ``docs/guide/workspace-architecture.md``.

MolExp persists every entity as a small JSON file alongside the payloads
it produces. This example seeds a realistic workspace and then prints
the on-disk tree with per-file sizes so you can see exactly where
catalog, manifest, artifact, log, and checkpoint bytes live.

Run directly::

    python examples/workspace/workspace_architecture.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me


async def task(ctx: me.RunContext) -> None:
    ctx.artifact.save("metrics.json", {"loss": 0.1})
    ctx.log("train").append("epoch 1 complete")
    ctx.checkpoint("epoch-1", data={"step": 1})


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-arch-"))
    ws = me.Workspace(root, name="arch-demo")
    project = ws.project("demo")
    exp = project.experiment("baseline")
    exp.set_workflow(task)
    run = exp.run()
    with run.start() as ctx:
        await exp.workflow.execute(run_context=ctx)

    print(f"workspace root: {root}\n")
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        print(f"  {rel}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
