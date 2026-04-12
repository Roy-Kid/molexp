"""Submission logic using molq types directly."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}


class SubmitHandler:
    """Stateful run handler that submits jobs via molq.

    Callable with signature ``(script, mol_run, exp_spec, project_spec)``.
    Accumulates :class:`~molq.JobHandle` objects so that :meth:`wait` can
    block until every submitted job reaches a terminal state.
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
        self._cluster = cluster
        self._res = _strip_none(resources)
        self._sched = _strip_none(scheduling)
        self._block = block
        self._handles: list[Any] = []
        self._submitor: Any = None

    # ------------------------------------------------------------------
    # Callable protocol

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

        if self._submitor is None:
            self._submitor = Submitor(
                cluster_name=self._cluster or "default",
                scheduler=self._scheduler,
            )

        res = self._res
        sched = self._sched
        run_dir = Path(mol_run.run_dir)
        # run_id is now 8 chars; no truncation needed.
        job_name = f"{project_spec.name[:20]}-{mol_run.id}"

        handle = self._submitor.submit(
            argv=[
                sys.executable,
                "-m",
                "molexp.plugins.submit_molq.worker",
                str(script.resolve()),
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
                output_file=str(run_dir / "job.out"),
                error_file=str(run_dir / "job.err"),
            ),
            metadata={
                "run_id": mol_run.id,
                "run_dir": str(run_dir),
            },
        )
        self._handles.append(handle)

        # Write scheduler job IDs back into run.json for cross-reference.
        # Enables: grep -r '"slurm_job_id": "..."' runs/*/run.json
        if hasattr(mol_run, "update_job_ids"):
            slurm_id = getattr(handle, "scheduler_job_id", None)
            molq_id = getattr(handle, "job_id", None)
            mol_run.update_job_ids(
                slurm_job_id=str(slurm_id) if slurm_id is not None else None,
                molq_job_id=str(molq_id) if molq_id is not None else None,
            )

    # ------------------------------------------------------------------
    # Blocking / monitoring

    def wait(self) -> None:
        """Block until all submitted jobs reach a terminal state.

        Displays a live Rich table that refreshes as job states change.
        No-op when no jobs have been submitted or *block* was not requested.
        """
        if not self._block or not self._handles or self._submitor is None:
            return

        import time

        from rich.console import Console
        from rich.live import Live
        from rich.table import Table

        _TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out", "lost"})
        _STATE_COLOR: dict[str, str] = {
            "succeeded": "green",
            "failed": "red",
            "cancelled": "yellow",
            "timed_out": "yellow",
            "lost": "red",
            "running": "green",
            "queued": "blue",
            "submitted": "cyan",
            "created": "dim",
        }

        job_ids = [h.job_id for h in self._handles]

        def _build_table(records: dict[str, Any]) -> Table:
            done = sum(
                1 for jid in job_ids
                if records.get(jid) is not None
                and records[jid].state.value in _TERMINAL
            )
            tbl = Table(title=f"Jobs — {done}/{len(job_ids)} done")
            tbl.add_column("Run", style="cyan", no_wrap=True)
            tbl.add_column("State", no_wrap=True)
            tbl.add_column("Scheduler ID", style="dim", no_wrap=True)
            for jid in job_ids:
                rec = records.get(jid)
                if rec is None:
                    tbl.add_row("—", "[dim]unknown[/dim]", "—")
                    continue
                state = rec.state.value
                color = _STATE_COLOR.get(state, "white")
                run_id = rec.metadata.get("run_id", jid[:8])
                sched_id = rec.scheduler_job_id or "—"
                tbl.add_row(run_id, f"[{color}]{state}[/{color}]", sched_id)
            return tbl

        console = Console()
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                self._submitor.refresh()
                records: dict[str, Any] = {}
                for jid in job_ids:
                    try:
                        records[jid] = self._submitor.get(jid)
                    except Exception:
                        pass
                live.update(_build_table(records))
                if all(
                    records.get(jid) is not None
                    and records[jid].state.value in _TERMINAL
                    for jid in job_ids
                ):
                    break
                time.sleep(2)


def make_submit_handler(
    *,
    scheduler: str,
    cluster: str | None,
    resources: dict[str, Any],
    scheduling: dict[str, Any],
<<<<<<< HEAD
    block: bool = False,
) -> SubmitHandler:
    """Return a :class:`SubmitHandler` configured for the given scheduler.

    The handler is callable with signature
    ``(script, mol_run, exp_spec, project_spec)`` and accumulates submitted
    job handles.  Call :meth:`SubmitHandler.wait` after dispatching all runs
    to block until every job finishes (only when *block* is ``True``).

    All ``None`` values in *resources* and *scheduling* are stripped so that
    molq passes them through as unset, letting each scheduler use its own
    defaults.
=======
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
>>>>>>> 0826146fa100cff8f2c688347f73526dc1d5aa8b
    """
    return SubmitHandler(
        scheduler=scheduler,
        cluster=cluster,
        resources=resources,
        scheduling=scheduling,
<<<<<<< HEAD
        block=block,
=======
>>>>>>> 0826146fa100cff8f2c688347f73526dc1d5aa8b
    )
