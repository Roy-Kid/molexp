"""Submission logic using molq types directly."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}


class SubmitHandler:
    """Callable run handler that submits each run via molq.

    Accumulates all submitted :class:`~molexp.workspace.run.Run` objects in
    :attr:`submitted_runs` so the caller can pass them to a monitor after the
    sweep completes.

    Args:
        scheduler: Scheduler type: ``"slurm"``, ``"pbs"``, or ``"lsf"``.
        cluster: molq cluster name (``None`` → ``"default"``).
        resources: Sparse dict of resource overrides (``None`` values stripped).
        scheduling: Sparse dict of scheduling overrides (``None`` values stripped).
    """

    def __init__(
        self,
        *,
        scheduler: str,
        cluster: str | None,
        resources: dict[str, Any],
        scheduling: dict[str, Any],
    ) -> None:
        self._scheduler  = scheduler
        self._cluster    = cluster or "default"
        self._resources  = _strip_none(resources)
        self._scheduling = _strip_none(scheduling)
        self.submitted_runs: list[Any] = []

    def __call__(
        self,
        script: Path,
        mol_run: Any,
        exp_spec: Any,
        project_spec: Any,
    ) -> None:
        from molq import (
            Duration,
            JobExecution,
            JobResources,
            JobScheduling,
            Memory,
            Submitor,
        )

        res   = self._resources
        sched = self._scheduling

        job_name = f"{project_spec.name[:20]}-{mol_run.id[:8]}"

        with Submitor(
            cluster_name=self._cluster,
            scheduler=self._scheduler,
        ) as submitor:
            submitor.submit(
                argv=[
                    sys.executable,
                    "-m",
                    "molexp.plugins.submit_molq.worker",
                    str(script.resolve()),
                    str(mol_run.run_dir),
                ],
                resources=JobResources(
                    cpu_count=res.get("cpus"),
                    memory=Memory.parse(res["mem"]) if res.get("mem") else None,
                    gpu_count=res.get("gpus"),
                    gpu_type=res.get("gpu_type"),
                    time_limit=Duration.parse(res["time"]) if res.get("time") else None,
                ),
                scheduling=JobScheduling(
                    queue=sched.get("queue"),
                    account=sched.get("account"),
                    qos=sched.get("qos"),
                ),
                execution=JobExecution(
                    job_name=job_name,
                    cwd=str(mol_run.run_dir),
                ),
                metadata={
                    "run_id":  mol_run.id,
                    "run_dir": str(mol_run.run_dir),
                },
            )

        self.submitted_runs.append(mol_run)


def make_submit_handler(
    *,
    scheduler: str,
    cluster: str | None,
    resources: dict[str, Any],
    scheduling: dict[str, Any],
) -> SubmitHandler:
    """Return a :class:`SubmitHandler` configured for the given scheduler.

    The handler is callable with the standard ``(script, mol_run, exp_spec,
    project_spec)`` signature used by :func:`~molexp.cli._execute_sweep`.
    After the sweep, :attr:`SubmitHandler.submitted_runs` contains every run
    that was successfully submitted.

    Args:
        scheduler: Scheduler backend: ``"slurm"``, ``"pbs"``, or ``"lsf"``.
        cluster: molq cluster name; ``None`` defaults to ``"default"``.
        resources: Resource options dict (``None`` values are stripped).
        scheduling: Scheduling options dict (``None`` values are stripped).

    Returns:
        Configured :class:`SubmitHandler` instance.
    """
    return SubmitHandler(
        scheduler=scheduler,
        cluster=cluster,
        resources=resources,
        scheduling=scheduling,
    )
