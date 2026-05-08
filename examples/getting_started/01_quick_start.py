"""Quick start — workspace + experiment + tracked run, end to end.

Matches ``docs/getting-started/quick-start.md``.

Run directly::

    python examples/getting_started/01_quick_start.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import molexp as me


async def train(ctx: me.RunContext) -> None:
    """User task body — runs inside one ``Run``."""
    lr = ctx.config.get("lr", 1e-3)
    epochs = ctx.config.get("epochs", 3)

    final_loss = 1.0 / (epochs * (lr * 1000 + 1))
    ctx.set_result("final_loss", final_loss)
    ctx.artifact.save("metrics.json", {"lr": lr, "epochs": epochs, "loss": final_loss})
    ctx.log("train").append(f"done lr={lr} epochs={epochs} loss={final_loss:.4f}")


async def main() -> None:
    workspace_root = Path(tempfile.mkdtemp(prefix="molexp-quickstart-"))
    print(f"workspace root: {workspace_root}")

    ws = me.Workspace(workspace_root, name="quickstart")
    project = ws.project("demo")
    experiment = project.experiment("train", params={"lr": 1e-3})
    experiment.set_workflow(train)

    run = experiment.run(parameters={"seed": 0})
    with run.start() as ctx:
        result = await experiment.workflow.execute(run_context=ctx)

    run_json = json.loads((run.run_dir / "run.json").read_text())
    print(f"status:     {result.status}")
    print(f"final_loss: {run_json['context']['results']['final_loss']}")
    print(f"run_dir:    {run.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
