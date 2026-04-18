"""Submission logic using molq types directly."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .metadata import build_executor_info


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}


class SubmitHandler:
    """Stateful run handler that submits jobs via molq.

    Accumulates all submitted :class:`~molexp.workspace.run.Run` objects in
    :attr:`submitted_runs` so the caller can pass them to a monitor after the
    sweep completes.

    Args:
        scheduler: molq scheduler backend name.
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
        block: bool = False,
    ) -> None:
        self._scheduler = scheduler
        self._cluster = cluster or "default"
        self._res = _strip_none(resources)
        self._sched = _strip_none(scheduling)
        self._block = block
        self._handles: list[Any] = []
        self._submitor: Any = None
        self.submitted_runs: list[Any] = []

    # ------------------------------------------------------------------
    # Callable protocol

    def __call__(
        self,
        mol_run: Any,
        experiment: Any,
        project: Any,
    ) -> None:
        from molq import (
            Duration,
            JobExecution,
            JobResources,
            JobScheduling,
            Memory,
            Submitor,
        )

        res   = self._res
        sched = self._sched

        job_name = f"{project.name[:20]}-{mol_run.id[:8]}"
        run_dir = Path(mol_run.run_dir)

        # molq stores its own manifest under execution/; the actual execution
        # sub-directories are created by RunContext when the job starts.
        exec_root = run_dir / "execution"
        exec_root.mkdir(parents=True, exist_ok=True)

        with Submitor(
            cluster_name=self._cluster,
            scheduler=self._scheduler,
            jobs_dir=str(exec_root),
        ) as submitor:
            job = submitor.submit(
                argv=[
                    sys.executable,
                    "-m",
                    "molexp.cli",
                    "execute",
                    str(run_dir),
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
                    cwd=str(run_dir),
                    output_file=str(run_dir / "stdout.log"),
                    error_file=str(run_dir / "stderr.log"),
                ),
                metadata={
                    "run_id":  mol_run.id,
                    "run_dir": str(run_dir),
                },
            )

        mol_run._update_metadata(
            executor_info=build_executor_info(
                scheduler=self._scheduler,
                cluster_name=self._cluster,
                job_id=job.job_id,
                scheduler_job_id=job.scheduler_job_id,
            )
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

    The handler is callable with the standard ``(script, mol_run, experiment,
    project)`` signature used by :func:`~molexp.cli._execute_sweep`.

    All ``None`` values in *resources* and *scheduling* are stripped so that
    molq passes them through as unset, letting each scheduler use its own
    defaults.

    Args:
        scheduler: molq scheduler backend name.
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
