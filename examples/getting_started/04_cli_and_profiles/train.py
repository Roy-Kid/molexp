"""Script that is meant to be driven by ``molexp run`` — not by ``python``.

Matches ``docs/getting-started/cli-and-profiles.md``.

The script declares a workspace, an experiment, and the workflow it runs via
the fluent chain ``ws.project(...).experiment(...).run(wf, params=...)`` —
that declaration is what the CLI discovers. Execute it with::

    molexp run examples/getting_started/04_cli_and_profiles/train.py --profile smoke
    molexp run examples/getting_started/04_cli_and_profiles/train.py --profile prod
    molexp run examples/getting_started/04_cli_and_profiles/train.py \\
        --profile smoke --override lr=5e-4

``molcfg.yaml`` in this directory defines the ``smoke`` and ``prod`` profiles.
"""

from __future__ import annotations

from pathlib import Path

import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler

# Workspace lives next to this script so repeated ``molexp run`` calls reuse it.
WORKSPACE_ROOT = Path(__file__).resolve().parent / "_workspace"

wf = WorkflowCompiler(name="train")


@wf.task
async def train(ctx: TaskContext) -> dict:
    """``ctx.config`` is the resolved molcfg profile selected via ``--profile``."""
    lr = ctx.config.get("lr", 1e-3)
    epochs = ctx.config.get("epochs", 1)
    final_loss = 1.0 / (epochs * (lr * 1000 + 1))
    return {
        "profile": ctx.config.name,
        "lr": lr,
        "epochs": epochs,
        "final_loss": final_loss,
    }


(
    me.Workspace(WORKSPACE_ROOT, name="cli-demo")
    .project("demo")
    .experiment("train")
    .run(wf.compile(), params={"seed": [0]})
)
