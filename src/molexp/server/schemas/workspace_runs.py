"""Schemas for the workspace-level Runs aggregator.

A flat row per run with embedded executions, suitable for a tree-style
table where each run row expands to show its execution attempts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from molexp.workspace import Run

from .molq import MolqJobSummary  # noqa: F401  (preserve import surface)


class WorkspaceExecutionRow(BaseModel):
    """One execution attempt of a run, surfaced for the workspace runs table."""

    executionId: str
    runId: str
    status: str
    startedAt: str
    finishedAt: str | None = None
    durationSeconds: float | None = None
    schedulerJobId: str | None = None
    backend: str | None = None
    backendMetadata: dict[str, str] = Field(default_factory=dict)


class WorkspaceRunRow(BaseModel):
    """One run, with its execution history nested for tree expansion."""

    id: str
    name: str
    projectId: str
    projectName: str
    experimentId: str
    experimentName: str
    status: str
    backend: str | None = None
    cluster: str | None = None
    scheduler: str | None = None
    target: str | None = None
    profile: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    createdAt: str
    finishedAt: str | None = None
    executionCount: int = 0
    latestSchedulerJobId: str | None = None
    executions: list[WorkspaceExecutionRow] = Field(default_factory=list)

    @classmethod
    def from_run(
        cls,
        run: Run,
        *,
        project_name: str,
        experiment_name: str,
    ) -> WorkspaceRunRow:
        executor = dict(run.metadata.executor_info or {})
        backend = executor.get("backend")
        cluster = executor.get("cluster_name")
        scheduler = executor.get("scheduler")

        executions = [
            _build_execution_row(run.id, rec, executor) for rec in run.metadata.execution_history
        ]
        latest_sched_id: str | None = None
        for rec in reversed(executions):
            if rec.schedulerJobId:
                latest_sched_id = rec.schedulerJobId
                break

        return cls(
            id=run.id,
            name=run.id,
            projectId=run.experiment.project.id,
            projectName=project_name,
            experimentId=run.experiment.id,
            experimentName=experiment_name,
            status=run.status,
            backend=backend,
            cluster=cluster,
            scheduler=scheduler,
            target=run.metadata.target,
            profile=run.metadata.profile,
            parameters=dict(run.parameters),
            createdAt=run.metadata.created_at.isoformat(),
            finishedAt=(
                run.metadata.finished_at.isoformat() if run.metadata.finished_at else None
            ),
            executionCount=len(executions),
            latestSchedulerJobId=latest_sched_id,
            executions=executions,
        )


def _build_execution_row(
    run_id: str,
    record: Any,
    executor_info: dict[str, Any],
) -> WorkspaceExecutionRow:
    started = record.started_at
    finished = record.finished_at
    duration: float | None = None
    if finished is not None:
        duration = max(0.0, (finished - started).total_seconds())

    backend_metadata: dict[str, str] = {}
    for key, value in executor_info.items():
        if key == "backend" or value is None:
            continue
        backend_metadata[str(key)] = str(value)
    if record.scheduler_job_id:
        backend_metadata["scheduler_job_id"] = record.scheduler_job_id

    return WorkspaceExecutionRow(
        executionId=record.execution_id,
        runId=run_id,
        status=record.status,
        startedAt=started.isoformat(),
        finishedAt=finished.isoformat() if finished else None,
        durationSeconds=duration,
        schedulerJobId=record.scheduler_job_id,
        backend=executor_info.get("backend"),
        backendMetadata=backend_metadata,
    )


class WorkspaceRunsStats(BaseModel):
    total: int = 0
    running: int = 0
    pending: int = 0
    failed: int = 0
    succeeded: int = 0


class WorkspaceRunsResponse(BaseModel):
    runs: list[WorkspaceRunRow]
    stats: WorkspaceRunsStats
    total: int
    truncated: bool = False


_RUNNING_STATES = {"running"}
_PENDING_STATES = {"pending", "queued", "submitted", "created"}
_FAILED_STATES = {"failed", "timed_out", "cancelled", "lost"}
_SUCCEEDED_STATES = {"succeeded"}


def compute_workspace_runs_stats(rows: list[WorkspaceRunRow]) -> WorkspaceRunsStats:
    stats = WorkspaceRunsStats(total=len(rows))
    for row in rows:
        state = row.status.lower()
        if state in _RUNNING_STATES:
            stats.running += 1
        elif state in _PENDING_STATES:
            stats.pending += 1
        elif state in _FAILED_STATES:
            stats.failed += 1
        elif state in _SUCCEEDED_STATES:
            stats.succeeded += 1
    return stats


__all__ = [
    "WorkspaceExecutionRow",
    "WorkspaceRunRow",
    "WorkspaceRunsStats",
    "WorkspaceRunsResponse",
    "compute_workspace_runs_stats",
]
