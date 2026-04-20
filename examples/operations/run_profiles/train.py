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


WORKSPACE_ROOT = Path(__file__).resolve().parent / "_workspace"


def train(ctx: me.RunContext) -> None:
    # The framework treats profile contents as opaque user data.
    # Read whatever fields your task needs, with sane defaults.
    epochs = ctx.config.get("epochs", 100)
    dataset = ctx.config.get("dataset", "qm9")
    lr = ctx.config.get("optimizer", {}).get("lr", 1e-3)
    batch = ctx.config.get("batch_size", 32)

    if ctx.config.get("skip_heavy_compute"):
        # Single "pretend to train" iteration so the example stays fast.
        ctx.set_result("mode", "lightweight")
    else:
        ctx.set_result("mode", "full")

    ctx.set_result("profile", ctx.config.name)
    ctx.set_result("epochs", epochs)
    ctx.set_result("dataset", dataset)
    ctx.set_result("lr", lr)
    ctx.set_result("batch_size", batch)
    ctx.log("train").append(
        f"profile={ctx.config.name} dataset={dataset} epochs={epochs} "
        f"batch={batch} lr={lr}"
    )


ws = me.Workspace(WORKSPACE_ROOT, name="run-profiles-demo")
project = ws.project("demo")
exp = project.experiment("train")
exp.set_workflow(train)

me.entry(ws)
