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
from molexp.workflow import WorkflowCompiler

WORKSPACE_ROOT = Path(__file__).resolve().parent / "_workspace"

wf = WorkflowCompiler(name="train")


@wf.task
async def train(
    dataset: str = "qm9",
    epochs: int = 100,
    batch_size: int = 32,
    optimizer: dict | None = None,
    skip_heavy_compute: bool = False,
) -> dict:
    # The framework treats profile contents as opaque user data. Each field a
    # task needs is declared as a named parameter and bound by name from the
    # run's build-time config (the merged molcfg profile), with a default so the
    # task also runs with no profile selected. A nested block like
    # ``optimizer: {lr: ...}`` arrives whole as the ``optimizer`` parameter. The
    # chosen profile name is recorded on the run record, not read here.
    lr = (optimizer or {}).get("lr", 1e-3)
    mode = "lightweight" if skip_heavy_compute else "full"

    return {
        "mode": mode,
        "dataset": dataset,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }


(
    me.Workspace(WORKSPACE_ROOT, name="run-profiles-demo")
    .project("demo")
    .experiment("train")
    .run(wf.compile(), params={"seed": [0]})
)
