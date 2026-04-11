"""Execution backend protocol and submission model.

Defines the :class:`ExecutionBackend` protocol that plugins implement
to submit runs to remote schedulers (SLURM, PBS, etc.).

The CLI delegates all remote submission through this protocol — it
never constructs scheduler-specific argv or imports scheduler libraries
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RunSubmission:
    """All information a backend needs to submit a single run.

    Attributes:
        script: Path to the user's experiment script.
        run_dir: Path to the run directory on the filesystem.
        run_id: Unique identifier of the run.
        experiment_name: User-facing experiment name.
        project_name: User-facing project name.
        resources: Backend-specific resource requests (gpus, mem, time, …).
    """

    script: Path
    run_dir: Path
    run_id: str
    experiment_name: str
    project_name: str
    resources: dict[str, Any]


@runtime_checkable
class ExecutionBackend(Protocol):
    """Plugin protocol for remote run submission.

    Backends control the full submission flow: argv construction,
    resource mapping, and scheduler interaction.  The CLI only calls
    :meth:`submit_run` with a :class:`RunSubmission`.
    """

    def submit_run(self, submission: RunSubmission) -> None:
        """Submit a single run for remote execution.

        Args:
            submission: Submission specification.
        """
        ...
