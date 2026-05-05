"""Read-only aggregation over molq Submitors for the Remote Operations UI.

Provides a thin facade that walks all configured molq profiles, fetches
:class:`~molq.JobRecord` snapshots, and shapes them into UI-friendly summaries.

This module does not mutate molq state. All scheduler-side operations
(submission, cancellation, retry) remain the job of
:mod:`molexp.plugins.submit_molq.submit`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from molq import JobNotFoundError, MolqError, Submitor
from molq.config import load_config
from molq.models import DependencyPreview, JobRecord, StatusTransition
from molq.status import JobState

# ── State buckets ──────────────────────────────────────────────────────────
# Map raw JobState → high-level bucket used by the overview's stat cards.
# Kept here so backend tests and the UI agree on bucket membership.

_PENDING_STATES = frozenset({JobState.CREATED, JobState.SUBMITTED, JobState.QUEUED})
_RUNNING_STATES = frozenset({JobState.RUNNING})
_FAILED_STATES = frozenset({JobState.FAILED, JobState.TIMED_OUT, JobState.CANCELLED, JobState.LOST})
_SUCCEEDED_STATES = frozenset({JobState.SUCCEEDED})

# 24h window for avg-wait sample selection.
_WAIT_WINDOW_SECONDS = 24 * 60 * 60

# Log tailing knobs.
_LOG_POLL_SECONDS = 0.25
_LOG_IDLE_TIMEOUT_SECONDS = 5 * 60
_LOG_READ_CHUNK = 16 * 1024


# ── DTOs (plain dataclasses, decoupled from API schemas) ──────────────────


@dataclass(frozen=True)
class TargetSummary:
    name: str
    scheduler: str
    cluster_name: str | None
    jobs_dir: str | None
    healthy: bool
    health_reason: str | None
    active_jobs: int


@dataclass(frozen=True)
class JobSummary:
    target: str
    job_id: str
    scheduler_job_id: str | None
    cluster_name: str | None
    scheduler: str | None
    name: str | None
    state: str
    submitted_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    exit_code: int | None
    duration_seconds: float | None
    cwd: str | None


@dataclass(frozen=True)
class JobTransition:
    timestamp: datetime
    from_state: str | None
    to_state: str
    reason: str | None


@dataclass(frozen=True)
class JobDetail:
    summary: JobSummary
    failure_reason: str | None
    metadata: dict[str, str]
    command_display: str | None
    transitions: tuple[JobTransition, ...]
    upstream_total: int
    upstream_satisfied: int
    downstream_total: int


@dataclass(frozen=True)
class QueueStats:
    running: int = 0
    pending: int = 0
    failed: int = 0
    succeeded: int = 0
    avg_wait_seconds: float | None = None


@dataclass(frozen=True)
class JobsPage:
    jobs: tuple[JobSummary, ...] = field(default_factory=tuple)
    stats: QueueStats = field(default_factory=QueueStats)


# ── Submitor cache ────────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _submitor_for(name: str, config_path: str | None = None) -> Submitor:
    """Return a process-cached :class:`Submitor` for *name*.

    The molq SQLite store opens a connection per Submitor instance, so caching
    by ``(profile_name, config_path)`` keeps the dashboard cheap when polled
    every few seconds.
    """
    return Submitor.from_profile(name, config_path=config_path)


def _reset_submitor_cache() -> None:
    """Test-only helper: drop any cached Submitor instances."""
    _submitor_for.cache_clear()


# ── Public read API ───────────────────────────────────────────────────────


def list_targets(config_path: str | Path | None = None) -> list[TargetSummary]:
    """Return one :class:`TargetSummary` per configured molq profile.

    Per-target failures (unreadable store, missing scheduler binary, etc.) are
    swallowed and surfaced as ``healthy=False`` rather than aborting the list.
    """
    config = load_config(config_path)
    cfg_str = str(config_path) if config_path is not None else None

    targets: list[TargetSummary] = []
    for name, profile in config.profiles.items():
        active = 0
        healthy = True
        reason: str | None = None
        try:
            submitor = _submitor_for(name, cfg_str)
            active = sum(1 for _ in submitor.list_jobs(include_terminal=False))
        except Exception as exc:  # noqa: BLE001 — third-party errors are opaque
            healthy = False
            reason = f"{type(exc).__name__}: {exc}"
        targets.append(
            TargetSummary(
                name=name,
                scheduler=profile.scheduler,
                cluster_name=profile.cluster_name,
                jobs_dir=profile.jobs_dir,
                healthy=healthy,
                health_reason=reason,
                active_jobs=active,
            )
        )
    return targets


def list_jobs(
    target: str | None = None,
    *,
    include_terminal: bool = True,
    limit: int = 200,
    config_path: str | Path | None = None,
) -> list[JobSummary]:
    """List jobs across one or all configured targets.

    Sorted by ``submitted_at`` descending; jobs without a submitted timestamp
    sort last.
    """
    if limit <= 0:
        return []

    cfg_str = str(config_path) if config_path is not None else None
    config = load_config(config_path)
    target_names: list[str]
    if target is not None:
        if target not in config.profiles:
            raise KeyError(f"Unknown molq profile: {target!r}")
        target_names = [target]
    else:
        target_names = list(config.profiles.keys())

    collected: list[JobSummary] = []
    for name in target_names:
        try:
            submitor = _submitor_for(name, cfg_str)
            records = submitor.list_jobs(include_terminal=include_terminal)
        except Exception:  # noqa: BLE001 — keep iterating other targets
            continue
        collected.extend(_to_summary(name, r) for r in records)

    collected.sort(
        key=lambda j: j.submitted_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return collected[:limit]


def get_job(
    target: str,
    job_id: str,
    *,
    config_path: str | Path | None = None,
) -> JobDetail:
    """Return a :class:`JobDetail` for ``(target, job_id)``.

    Raises:
        KeyError: ``target`` is not a configured profile.
        JobNotFoundError: job not found in the store.
    """
    cfg_str = str(config_path) if config_path is not None else None
    config = load_config(config_path)
    if target not in config.profiles:
        raise KeyError(f"Unknown molq profile: {target!r}")

    submitor = _submitor_for(target, cfg_str)
    record = submitor.get_job(job_id)  # raises JobNotFoundError
    transitions = submitor.get_transitions(job_id)

    upstream_total = upstream_satisfied = downstream_total = 0
    try:
        preview = submitor.get_dependency_preview(job_id)
    except MolqError:
        preview = None
    if isinstance(preview, DependencyPreview):
        upstream_total = preview.upstream_total
        upstream_satisfied = preview.upstream_satisfied
        downstream_total = preview.downstream_total

    summary = _to_summary(target, record)
    return JobDetail(
        summary=summary,
        failure_reason=record.failure_reason,
        metadata=dict(record.metadata),
        command_display=record.command_display or None,
        transitions=tuple(_to_transition(t) for t in transitions),
        upstream_total=upstream_total,
        upstream_satisfied=upstream_satisfied,
        downstream_total=downstream_total,
    )


def compute_stats(jobs: list[JobSummary]) -> QueueStats:
    """Bucket *jobs* into running/pending/failed/succeeded counts plus avg wait.

    ``avg_wait_seconds`` is the mean of ``started_at - submitted_at`` over jobs
    where both timestamps are present and ``submitted_at`` falls within the last
    24 hours. Returns ``None`` when no eligible jobs exist (avoids a misleading
    ``0.0`` for an empty queue).
    """
    running = pending = failed = succeeded = 0
    waits: list[float] = []
    cutoff = datetime.now(timezone.utc).timestamp() - _WAIT_WINDOW_SECONDS

    for job in jobs:
        try:
            state = JobState(job.state)
        except ValueError:
            continue
        if state in _RUNNING_STATES:
            running += 1
        elif state in _PENDING_STATES:
            pending += 1
        elif state in _FAILED_STATES:
            failed += 1
        elif state in _SUCCEEDED_STATES:
            succeeded += 1

        if (
            job.submitted_at is not None
            and job.started_at is not None
            and job.submitted_at.timestamp() >= cutoff
        ):
            delta = (job.started_at - job.submitted_at).total_seconds()
            if delta >= 0:
                waits.append(delta)

    avg_wait = mean(waits) if waits else None
    return QueueStats(
        running=running,
        pending=pending,
        failed=failed,
        succeeded=succeeded,
        avg_wait_seconds=avg_wait,
    )


def fetch_page(
    target: str | None,
    *,
    include_terminal: bool = True,
    limit: int = 200,
    config_path: str | Path | None = None,
) -> JobsPage:
    """Convenience wrapper: list jobs and compute stats together."""
    jobs = list_jobs(
        target,
        include_terminal=include_terminal,
        limit=limit,
        config_path=config_path,
    )
    return JobsPage(jobs=tuple(jobs), stats=compute_stats(jobs))


async def tail_log(
    target: str,
    job_id: str,
    *,
    stream: str = "stdout",
    config_path: str | Path | None = None,
) -> AsyncIterator[str]:
    """Yield newline-terminated chunks from a job's stdout/stderr file.

    Stops when the job reaches a terminal state and its log has been fully
    drained, or after :data:`_LOG_IDLE_TIMEOUT_SECONDS` of inactivity. Yields
    a final ``"[stream closed]"`` line so subscribers know the stream ended
    deliberately rather than from a network drop.
    """
    if stream not in ("stdout", "stderr"):
        raise ValueError(f"stream must be 'stdout' or 'stderr', got {stream!r}")

    cfg_str = str(config_path) if config_path is not None else None
    submitor = _submitor_for(target, cfg_str)
    record = submitor.get_job(job_id)  # raises JobNotFoundError

    log_path = _resolve_log_path(record, stream)
    idle_for = 0.0

    # Wait for the file to appear if the job hasn't started writing yet.
    while not log_path.exists():
        if idle_for >= _LOG_IDLE_TIMEOUT_SECONDS:
            yield "[stream closed]"
            return
        await asyncio.sleep(_LOG_POLL_SECONDS)
        idle_for += _LOG_POLL_SECONDS

    # Tail the file. Use a thread to avoid blocking the event loop on slow
    # network filesystems where readers can stall on read(2).
    def _open_and_seek() -> Any:
        f = log_path.open("r", encoding="utf-8", errors="replace")
        f.seek(0)
        return f

    fh = await asyncio.to_thread(_open_and_seek)
    idle_for = 0.0
    try:
        while True:
            chunk = await asyncio.to_thread(fh.read, _LOG_READ_CHUNK)
            if chunk:
                idle_for = 0.0
                # Emit one event per line so the SSE consumer can render
                # incrementally; preserve terminal lacking a newline.
                for line in chunk.splitlines():
                    if line:
                        yield line
                continue

            # No new bytes. Decide whether to keep waiting or stop.
            try:
                fresh = submitor.get_job(job_id)
            except JobNotFoundError:
                break
            if JobState(fresh.state).is_terminal:
                break
            if idle_for >= _LOG_IDLE_TIMEOUT_SECONDS:
                break
            await asyncio.sleep(_LOG_POLL_SECONDS)
            idle_for += _LOG_POLL_SECONDS
    finally:
        await asyncio.to_thread(fh.close)

    yield "[stream closed]"


# ── Internal helpers ──────────────────────────────────────────────────────


def _to_summary(target: str, record: JobRecord) -> JobSummary:
    return JobSummary(
        target=target,
        job_id=record.job_id,
        scheduler_job_id=record.scheduler_job_id,
        cluster_name=record.cluster_name or None,
        scheduler=record.scheduler or None,
        name=record.metadata.get("job_name") or record.metadata.get("name"),
        state=record.state.value,
        submitted_at=_to_datetime(record.submitted_at),
        started_at=_to_datetime(record.started_at),
        finished_at=_to_datetime(record.finished_at),
        exit_code=record.exit_code,
        duration_seconds=_duration_seconds(record),
        cwd=record.cwd or None,
    )


def _to_transition(t: StatusTransition) -> JobTransition:
    return JobTransition(
        timestamp=_to_datetime(t.timestamp) or datetime.now(timezone.utc),
        from_state=t.old_state.value if t.old_state is not None else None,
        to_state=t.new_state.value,
        reason=t.reason,
    )


def _to_datetime(ts: float | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _duration_seconds(record: JobRecord) -> float | None:
    start = record.started_at
    end = record.finished_at
    if start is None:
        return None
    if end is None:
        # Running: report elapsed since start.
        end = datetime.now(timezone.utc).timestamp()
    delta = end - start
    return delta if delta >= 0 else None


def _resolve_log_path(record: JobRecord, stream: str) -> Path:
    """Locate the on-disk log file for *record*.

    molexp's :class:`~molexp.plugins.submit_molq.submit.SubmitHandler` writes
    ``stdout.log`` / ``stderr.log`` directly under the job's ``cwd`` (the
    per-execution directory). For jobs submitted by other tooling we fall back
    to ``cwd/.molq/jobs/<job_id>/{stream}.log``.
    """
    cwd = Path(record.cwd) if record.cwd else Path.cwd()
    primary = cwd / f"{stream}.log"
    if primary.exists():
        return primary
    fallback = cwd / ".molq" / "jobs" / record.job_id / f"{stream}.log"
    return fallback if fallback.exists() else primary


__all__ = [
    "TargetSummary",
    "JobSummary",
    "JobTransition",
    "JobDetail",
    "QueueStats",
    "JobsPage",
    "list_targets",
    "list_jobs",
    "get_job",
    "compute_stats",
    "fetch_page",
    "tail_log",
]
