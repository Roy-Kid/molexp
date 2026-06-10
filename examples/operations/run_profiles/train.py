"""Run profiles in anger — same script, three execution shapes.

Matches ``docs/guide/run-profiles.md``.

Execute with the CLI, picking one of the profiles defined in
``molcfg.yaml``::

    molexp run examples/operations/run_profiles/train.py --profile dry-run
    molexp run examples/operations/run_profiles/train.py --profile smoke
    molexp run examples/operations/run_profiles/train.py --profile large-batch

Override a single field without editing the config file::

    molexp run examples/operations/run_profiles/train.py \\
        --profile smoke --override optimizer.lr=0.0005

Resume a failed or cancelled run under the same profile::

    molexp run examples/operations/run_profiles/train.py --profile smoke --resume
"""

from __future__ import annotations

from pathlib import Path

import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler

WORKSPACE_ROOT = Path(__file__).resolve().parent / "_workspace"

wf = WorkflowCompiler(name="train")


@wf.task
async def train(ctx: TaskContext) -> dict:
    # The framework treats profile contents as opaque user data.
    # Read whatever fields your task needs, with sane defaults.
    epochs = ctx.config.get("epochs", 100)
    dataset = ctx.config.get("dataset", "qm9")
    lr = ctx.config.get("optimizer", {}).get("lr", 1e-3)
    batch = ctx.config.get("batch_size", 32)
    mode = "lightweight" if ctx.config.get("skip_heavy_compute") else "full"

    return {
        "profile": ctx.config.name,
        "mode": mode,
        "dataset": dataset,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch,
    }


(
    me.Workspace(WORKSPACE_ROOT, name="run-profiles-demo")
    .project("demo")
    .experiment("train")
    .run(wf.compile(), params={"seed": [0]})
)
