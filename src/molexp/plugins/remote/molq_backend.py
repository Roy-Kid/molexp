"""Molq-based execution backend for SLURM submission.

Implements the :class:`~molexp.plugins.remote.backend.ExecutionBackend`
protocol using ``molq.Submitor``.
"""

from __future__ import annotations

from typing import Any

from molexp.plugins.remote.backend import RunSubmission


class MolqBackend:
    """SLURM execution backend powered by molq.

    The backend controls argv construction and scheduler interaction.
    The CLI never imports molq directly.

    Args:
        cluster: molq cluster name.
        scheduler: Scheduler type (default ``"slurm"``).
        scheduler_options: Extra scheduler-specific options.
    """

    def __init__(
        self,
        cluster: str = "hpc",
        scheduler: str = "slurm",
        scheduler_options: Any = None,
    ) -> None:
        self.cluster = cluster
        self.scheduler = scheduler
        self.scheduler_options = scheduler_options

    def submit_run(self, submission: RunSubmission) -> None:
        """Submit a single run to SLURM via molq."""
        from molq import (
            Duration,
            JobExecution,
            JobResources,
            JobScheduling,
            Memory,
            Submitor,
        )
        from molq.options import SlurmSchedulerOptions

        res = submission.resources
        job_name = f"{submission.project_name[:20]}-{submission.run_id[:8]}"

        opts = self.scheduler_options or SlurmSchedulerOptions()
        submitor = Submitor(
            cluster_name=self.cluster,
            scheduler=self.scheduler,
            scheduler_options=opts,
        )

        argv = self._build_argv(submission)

        submitor.submit(
            argv=argv,
            resources=JobResources(
                cpu_count=res.get("cpus", 8),
                memory=Memory.parse(res.get("mem", "40G")),
                gpu_count=res.get("gpus", 1),
                gpu_type=res.get("gpu_type"),
                time_limit=Duration.parse(res.get("time", "12:00:00")),
            ),
            scheduling=JobScheduling(
                queue=res.get("partition", "gpu"),
                account=res.get("account"),
                qos=res.get("qos"),
            ),
            execution=JobExecution(
                job_name=job_name,
                cwd=str(submission.run_dir),
                output_file=str(submission.run_dir / "slurm_%j.out"),
                error_file=str(submission.run_dir / "slurm_%j.err"),
            ),
            metadata={
                "run_id": submission.run_id,
                "run_dir": str(submission.run_dir),
            },
        )

    def _build_argv(self, submission: RunSubmission) -> list[str]:
        """Build the re-invocation command line.

        Uses ``python -m molexp.plugins.remote.worker`` — a self-contained
        entry point that imports the script, finds the workflow, and
        executes the run.  No CLI command needed.
        """
        import sys

        return [
            sys.executable,
            "-m",
            "molexp.plugins.remote.worker",
            str(submission.script.resolve()),
            str(submission.run_dir),
        ]
