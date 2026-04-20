"""Script that is meant to be driven by ``molexp run`` — not by ``python``.

Matches ``docs/getting-started/cli-and-profiles.md``.

The script constructs a workspace and experiment, registers them with
``me.entry(ws)``, and leaves execution to the CLI. Run it with::

    molexp run examples/getting_started/04_cli_and_profiles/train.py --profile smoke
    molexp run examples/getting_started/04_cli_and_profiles/train.py --profile prod
    molexp run examples/getting_started/04_cli_and_profiles/train.py \\
        --profile smoke --override lr=5e-4

``molcfg.yaml`` in this directory defines the ``smoke`` and ``prod`` profiles.
"""

from __future__ import annotations

from pathlib import Path

import molexp as me

# Workspace lives next to this script so repeated ``molexp run`` calls reuse it.
WORKSPACE_ROOT = Path(__file__).resolve().parent / "_workspace"


def train(ctx: me.RunContext) -> None:
    lr = ctx.config.get("lr", 1e-3)
    epochs = ctx.config.get("epochs", 1)
    final_loss = 1.0 / (epochs * (lr * 1000 + 1))

    ctx.set_result("profile", ctx.config.name)
    ctx.set_result("lr", lr)
    ctx.set_result("epochs", epochs)
    ctx.set_result("final_loss", final_loss)
    ctx.log("train").append(
        f"profile={ctx.config.name} lr={lr} epochs={epochs} loss={final_loss:.4f}"
    )


ws = me.Workspace(WORKSPACE_ROOT, name="cli-demo")
project = ws.project("demo")
exp = project.experiment("train")
exp.set_workflow(train)

me.entry(ws)
