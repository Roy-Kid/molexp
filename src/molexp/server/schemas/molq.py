"""Response schemas for the molq Remote Operations endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from molexp.plugins.submit_molq.dashboard import (
    JobDetail,
    JobsPage,
    JobSummary,
    QueueStats,
    TargetSummary,
)


class MolqTargetSummary(BaseModel):
    name: str
    scheduler: str
    clusterName: str | None = None
    jobsDir: str | None = None
    healthy: bool
    healthReason: str | None = None
    activeJobs: int

    @classmethod
    def from_dataclass(cls, target: TargetSummary) -> MolqTargetSummary:
        return cls(
            name=target.name,
            scheduler=target.scheduler,
            clusterName=target.cluster_name,
            jobsDir=target.jobs_dir,
            healthy=target.healthy,
            healthReason=target.health_reason,
            activeJobs=target.active_jobs,
        )


class MolqTargetListResponse(BaseModel):
    targets: list[MolqTargetSummary]
    total: int


class MolqJobSummary(BaseModel):
    target: str
    jobId: str
    schedulerJobId: str | None = None
    clusterName: str | None = None
    scheduler: str | None = None
    name: str | None = None
    state: str
    submittedAt: datetime | None = None
    startedAt: datetime | None = None
    finishedAt: datetime | None = None
    exitCode: int | None = None
    durationSeconds: float | None = None
    cwd: str | None = None

    @classmethod
    def from_dataclass(cls, job: JobSummary) -> MolqJobSummary:
        return cls(
            target=job.target,
            jobId=job.job_id,
            schedulerJobId=job.scheduler_job_id,
            clusterName=job.cluster_name,
            scheduler=job.scheduler,
            name=job.name,
            state=job.state,
            submittedAt=job.submitted_at,
            startedAt=job.started_at,
            finishedAt=job.finished_at,
            exitCode=job.exit_code,
            durationSeconds=job.duration_seconds,
            cwd=job.cwd,
        )


class MolqQueueStats(BaseModel):
    running: int = 0
    pending: int = 0
    failed: int = 0
    succeeded: int = 0
    avgWaitSeconds: float | None = None

    @classmethod
    def from_dataclass(cls, stats: QueueStats) -> MolqQueueStats:
        return cls(
            running=stats.running,
            pending=stats.pending,
            failed=stats.failed,
            succeeded=stats.succeeded,
            avgWaitSeconds=stats.avg_wait_seconds,
        )


class MolqJobsResponse(BaseModel):
    jobs: list[MolqJobSummary]
    stats: MolqQueueStats
    total: int

    @classmethod
    def from_page(cls, page: JobsPage) -> MolqJobsResponse:
        jobs = [MolqJobSummary.from_dataclass(j) for j in page.jobs]
        return cls(
            jobs=jobs,
            stats=MolqQueueStats.from_dataclass(page.stats),
            total=len(jobs),
        )


class MolqJobTransition(BaseModel):
    timestamp: datetime
    fromState: str | None = None
    toState: str
    reason: str | None = None


class MolqJobDetailResponse(MolqJobSummary):
    failureReason: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    commandDisplay: str | None = None
    transitions: list[MolqJobTransition] = Field(default_factory=list)
    upstreamTotal: int = 0
    upstreamSatisfied: int = 0
    downstreamTotal: int = 0

    @classmethod
    def from_dataclass(cls, detail: JobDetail) -> MolqJobDetailResponse:  # ty: ignore[invalid-method-override]
        base = MolqJobSummary.from_dataclass(detail.summary).model_dump()
        return cls(
            **base,
            failureReason=detail.failure_reason,
            metadata=detail.metadata,
            commandDisplay=detail.command_display,
            transitions=[
                MolqJobTransition(
                    timestamp=t.timestamp,
                    fromState=t.from_state,
                    toState=t.to_state,
                    reason=t.reason,
                )
                for t in detail.transitions
            ],
            upstreamTotal=detail.upstream_total,
            upstreamSatisfied=detail.upstream_satisfied,
            downstreamTotal=detail.downstream_total,
        )


__all__ = [
    "MolqJobDetailResponse",
    "MolqJobSummary",
    "MolqJobTransition",
    "MolqJobsResponse",
    "MolqQueueStats",
    "MolqTargetListResponse",
    "MolqTargetSummary",
]
