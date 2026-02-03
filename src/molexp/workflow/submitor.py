"""Submitor protocol for external task execution."""

from __future__ import annotations

from typing import Protocol, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from molq.resources import JobSpec
    from molq.jobstatus import JobStatus


class SubmitorProtocol(Protocol):
    """Protocol for job submission backends.

    This protocol defines the interface that submitors must implement
    to be used with WorkflowEngine. It matches molq's BaseSubmitor interface.

    Methods:
        submit: Submit a job and return job ID
        query: Query job status
        cancel: Cancel a running job
    """

    def submit(self, config: dict | "JobSpec") -> int:
        """Submit a job.

        Args:
            config: Job configuration as dict or JobSpec

        Returns:
            Job ID assigned by the scheduler
        """
        ...

    def query(self, job_id: int | None = None) -> dict[int, "JobStatus"]:
        """Query job status from the scheduler.

        Args:
            job_id: Specific job ID or None for all jobs

        Returns:
            Mapping of job IDs to JobStatus
        """
        ...

    def cancel(self, job_id: int) -> None:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel
        """
        ...
