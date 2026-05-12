"""How ``--scheduler slurm`` composes a ``SubmitHandler`` under the hood.

Matches ``docs/guide/molq.md``.

This example is explanatory rather than executable on its own: actual
submission requires a live scheduler. The code here builds the same
``SubmitHandler`` object that ``molexp run --scheduler slurm`` would
build, prints the worker command it would submit for a given run, and
prints the normalised executor metadata that would be written back to
``run.json``.

Run directly::

    python examples/operations/molq.py

For a real cluster submission, use the CLI::

    molexp run train.py --scheduler slurm \\
        --partition gpu --gpus 1 --cpus 8 --time 4h
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import molexp as me
from molexp.plugins.submit_molq.metadata import (
    build_executor_info,
    supported_schedulers,
)
from molexp.plugins.submit_molq.submit import SubmitHandler


def main() -> None:
    print(f"installed molq backends: {supported_schedulers()}\n")

    root = Path(tempfile.mkdtemp(prefix="molexp-molq-"))
    ws = me.Workspace(root, name="molq-demo")
    project = ws.add_project("demo")
    exp = project.add_experiment("train")
    run = exp.add_run(parameters={"seed": 0})

    # The CLI composes this same object from --scheduler/--cpus/--gpus/… .
    # Demonstration only; the CLI builds and uses this handler at submit time.
    _handler = SubmitHandler(
        scheduler="slurm",
        cluster=None,
        resources={"cpus": 8, "gpus": 1, "mem": "32G", "time": "4h"},
        scheduling={"queue": "gpu", "account": "myacct"},
    )

    # The worker command the plugin would submit for this run.
    cmd = [sys.executable, "-m", "molexp.cli", "execute", str(run.run_dir)]
    print("the plugin would submit:")
    print(f"  argv = {cmd}")

    # Normalised executor metadata that would be written back to run.json.
    executor_info = build_executor_info(
        scheduler="slurm",
        cluster_name="default",
        job_id="fake-0001",
        scheduler_job_id="slurm-123456",
    )
    print(f"\nrun.metadata.executor_info = {executor_info}")


if __name__ == "__main__":
    main()
