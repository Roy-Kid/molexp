"""Profile inheritance and ``--override`` dot-notation — ``molcfg.yaml`` advanced patterns.

Matches ``docs/guide/run-profiles.md``.

Demonstrates:

1. ``extends: base`` — profile inheritance in ``molcfg.yaml``.
2. ``--override`` dot notation — ``optimizer.lr=0.0005`` on the CLI.
3. Profile-resolved ``config_hash`` recorded on the run.
4. Task parameter binding from nested profile keys.

Build-time config fields from ``molcfg.yaml`` bind to task parameters by name;
each falls back to its declared default when the active profile omits it.

Run with::

    molexp run examples/operations/run_profiles_advanced/train.py --profile smoke
    molexp run examples/operations/run_profiles_advanced/train.py --profile production --override optimizer.lr=0.0005
"""

from __future__ import annotations

from pathlib import Path

import molexp as me
from molexp.workflow import WorkflowCompiler

WORKSPACE_ROOT = Path(__file__).resolve().parent / "_workspace"

wf = WorkflowCompiler(name="train")


@wf.task
def train(
    epochs: int = 10,
    optimizer: dict | None = None,
    skip_heavy_compute: bool = False,
) -> float:
    """Root — profile fields bind by name."""
    lr = (optimizer or {}).get("lr", 0.001)
    if skip_heavy_compute:
        return 0.0
    return float(epochs) * lr


(
    me.Workspace(WORKSPACE_ROOT, name="profiles-advanced")
    .project("demo")
    .experiment("train")
    .run(wf.compile(), params={"epochs": [3, 10]})
)
