"""Submission logic using molq types directly."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}


def make_submit_handler(
    *,
    scheduler: str,
    cluster: str | None,
    resources: dict[str, Any],
    scheduling: dict[str, Any],
) -> Callable[[Path, Any, Any, Any], None]:
    """Return a run handler that submits via molq.

    The handler signature is ``(script, mol_run, exp_spec, project_spec)``.
    All ``None`` values in *resources* and *scheduling* are stripped so that
    molq passes them through as unset, letting each scheduler use its own
    defaults.
    """
    res = _strip_none(resources)
    sched = _strip_none(scheduling)

    def handler(
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

        submitor = Submitor(
            cluster_name=cluster or "default",
            scheduler=scheduler,
        )
        job_name = f"{project_spec.name[:20]}-{mol_run.id[:8]}"

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
                "run_id": mol_run.id,
                "run_dir": str(mol_run.run_dir),
            },
        )

    return handler
