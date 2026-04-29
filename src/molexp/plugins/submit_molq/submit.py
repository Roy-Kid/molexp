"""Submission logic using molq types directly."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .metadata import build_executor_info

if TYPE_CHECKING:
    from molexp.workspace import ComputeTarget


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}


class SubmitHandler:
    """Stateful run handler that submits jobs via molq.

    Accumulates all submitted :class:`~molexp.workspace.run.Run` objects in
    :attr:`submitted_runs` so the caller can pass them to a monitor after the
    sweep completes.

    Args:
        scheduler: molq scheduler backend name. Ignored when *target* is set
            (the target carries its own scheduler choice).
        cluster: molq cluster name (``None`` → ``"default"``).
        resources: Sparse dict of resource overrides (``None`` values stripped).
        scheduling: Sparse dict of scheduling overrides (``None`` values stripped).
        target: When provided, jobs are routed through the target's transport
            and scheduler; the run dir is staged in before submit and staged
            out on terminal events.  When ``None`` the handler keeps the
            legacy local/in-process dispatch path.
    """

    def __init__(
        self,
        *,
        scheduler: str,
        cluster: str | None,
        resources: dict[str, Any],
        scheduling: dict[str, Any],
        block: bool = False,
        target: ComputeTarget | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._cluster = cluster or "default"
        self._res = _strip_none(resources)
        self._sched = _strip_none(scheduling)
        self._block = block
        self._target = target
        self._handles: list[Any] = []
        self._submitor: Any = None
        self.submitted_runs: list[Any] = []

    # ------------------------------------------------------------------
    # Callable protocol

    def __call__(
        self,
        _script: Any,
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

        from molexp.workflow._pydantic_graph.runtime import _make_execution_id

        res = self._res
        sched = self._sched
        target = self._target

        job_name = f"{project.name[:20]}-{mol_run.id[:8]}"
        run_dir = Path(mol_run.run_dir)

        # Pre-allocate the execution_id the worker will use.  When running
        # locally the per-attempt directory is created here so molq's
        # stdout/stderr/jobs paths land alongside the workflow.json the
        # worker will write.  When running on a remote target the staging
        # step takes care of mirror-creating the equivalent directory on
        # the transport's filesystem.
        execution_id = _make_execution_id(mol_run.id, run_dir)
        local_exec_dir = run_dir / "executions" / execution_id
        local_exec_dir.mkdir(parents=True, exist_ok=True)

        if target is None:
            # Legacy path — unchanged.
            transport = None
            scheduler_name = self._scheduler
            target_run_dir_ = str(run_dir)
            target_exec_dir = str(local_exec_dir)
            stage_out_cb = None
        else:
            from molexp.workspace.targets import target_run_dir, to_transport

            from .staging import stage_in, stage_out

            transport = to_transport(target)
            scheduler_name = target.scheduler
            target_run_dir_ = target_run_dir(target, project.workspace, mol_run)
            target_exec_dir = f"{target_run_dir_}/executions/{execution_id}"
            transport.mkdir(target_exec_dir, parents=True, exist_ok=True)
            stage_in(transport, mol_run, target)

            def stage_out_cb(_event: Any) -> None:
                stage_out(transport, mol_run, target, execution_id)  # type: ignore[arg-type]

        jobs_dir = f"{target_exec_dir}/jobs"
        # mkdir for the local case is implicit in Submitor; for remote case
        # we already created target_exec_dir, but we still need the jobs
        # subdir to exist before molq writes its manifest.
        if transport is not None:
            transport.mkdir(jobs_dir, parents=True, exist_ok=True)
        else:
            Path(jobs_dir).mkdir(parents=True, exist_ok=True)

        with Submitor(
            cluster_name=self._cluster,
            scheduler=scheduler_name,
            jobs_dir=jobs_dir,
            transport=transport,
        ) as submitor:
            if stage_out_cb is not None:
                from molq.callbacks import EventType

                for evt in (
                    EventType.JOB_COMPLETED,
                    EventType.JOB_FAILED,
                    EventType.JOB_CANCELLED,
                ):
                    submitor._event_bus.on(evt, stage_out_cb)

            job = submitor.submit(
                argv=[
                    sys.executable,
                    "-m",
                    "molexp.cli",
                    "execute",
                    target_run_dir_,
                    "--execution-id",
                    execution_id,
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
                    cwd=target_exec_dir,
                    output_file=f"{target_exec_dir}/stdout.log",
                    error_file=f"{target_exec_dir}/stderr.log",
                ),
                metadata={
                    "run_id": mol_run.id,
                    "run_dir": target_run_dir_,
                    "execution_id": execution_id,
                },
            )

        mol_run._update_metadata(
            executor_info=build_executor_info(
                scheduler=scheduler_name,
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
    target: ComputeTarget | None = None,
) -> SubmitHandler:
    """Return a :class:`SubmitHandler` configured for the given scheduler.

    The handler is callable with the standard ``(script, mol_run, experiment,
    project)`` signature used by :func:`~molexp.cli._execute_sweep`.  The
    leading ``script`` is accepted for uniformity with the local dispatcher
    and intentionally ignored; the worker rebuilds the run from ``run_dir``.

    All ``None`` values in *resources* and *scheduling* are stripped so that
    molq passes them through as unset, letting each scheduler use its own
    defaults.

    Args:
        scheduler: molq scheduler backend name.
        cluster: molq cluster name; ``None`` defaults to ``"default"``.
        resources: Resource options dict (``None`` values are stripped).
        scheduling: Scheduling options dict (``None`` values are stripped).
        target: Optional :class:`~molexp.workspace.ComputeTarget` — when set,
            jobs route through the target's transport + scheduler and the run
            dir is staged in/out across the transport.

    Returns:
        Configured :class:`SubmitHandler` instance.
    """
    return SubmitHandler(
        scheduler=scheduler,
        cluster=cluster,
        resources=resources,
        scheduling=scheduling,
        target=target,
    )
