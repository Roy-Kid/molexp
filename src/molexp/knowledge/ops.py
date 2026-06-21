"""Typed models for the ``_ops/`` operational sidecar (okf-04).

A Concept's ``_ops/`` dir holds **hot machine state** — physically isolated
from the knowledge-layer ``meta.yaml`` (per the OKF rewrite's identity-vs-
runtime split). This module types the Run-level operational document
(``_ops/run.json``): status, ownership, timestamps, heartbeat, and the
execution history. Models are frozen ``pydantic.BaseModel`` (rewritten whole
by atomic writes); the spellings mirror ``molexp.workspace.models``
(``RunStatus`` / ``RETRYABLE_STATUSES`` / ``ExecutionRecord``) so the eventual
migration is a straight swap.

Timestamps are **aware-UTC** (:func:`_utcnow`): heartbeat staleness is a
cross-host comparison, so a timezone is mandatory. (Contrast ``Folder``'s
``log.md`` history, which is naive-local because it is human-facing.)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel

#: Name of the Run operational document → ``_ops/run.json``.
RUN_OPS_NAME = "run"

#: Cadence at which a live run re-stamps its heartbeat (seconds).
HEARTBEAT_INTERVAL_SECONDS = 30.0

#: A heartbeat older than this is considered stale (seconds; ~10 min).
HEARTBEAT_STALE_SECONDS = 600.0


def _utcnow() -> datetime:
    """Return the current time as an aware-UTC :class:`datetime`."""
    return datetime.now(UTC)


class RunStatus(StrEnum):
    """Lifecycle status of a Run (mirrors ``workspace.models.RunStatus``)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Statuses from which ``resume`` / ``rerun`` apply.
RETRYABLE_STATUSES: frozenset[str] = frozenset({RunStatus.FAILED.value, RunStatus.CANCELLED.value})

#: Terminal statuses — a finished run carries a ``finished_at``.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {RunStatus.SUCCEEDED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)


class ExecutionRecord(BaseModel, frozen=True):
    """One physical attempt to execute a Run (mirrors workspace's record)."""

    execution_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    scheduler_job_id: str | None = None


class RunOpsState(BaseModel, frozen=True):
    """The Run-level operational document persisted at ``_ops/run.json``."""

    status: RunStatus = RunStatus.PENDING
    owner_pid: int | None = None
    owner_host: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    heartbeat_at: datetime | None = None
    current_execution_id: str | None = None
    executions: tuple[ExecutionRecord, ...] = ()

    @property
    def is_retryable(self) -> bool:
        """Whether ``resume`` / ``rerun`` apply (status in retryable set)."""
        return self.status in RETRYABLE_STATUSES

    def heartbeat_age(self, now: datetime) -> timedelta | None:
        """Time since the last heartbeat, or ``None`` if never beaten."""
        if self.heartbeat_at is None:
            return None
        return now - self.heartbeat_at

    def is_heartbeat_stale(
        self, now: datetime, *, threshold_seconds: float = HEARTBEAT_STALE_SECONDS
    ) -> bool:
        """Whether the heartbeat is older than *threshold_seconds*.

        A never-beaten state is **not** stale (there is no heartbeat to age);
        deciding what to do with such a run is reaping policy (okf-05), not
        this predicate.
        """
        age = self.heartbeat_age(now)
        if age is None:
            return False
        return age.total_seconds() > threshold_seconds


__all__ = [
    "HEARTBEAT_INTERVAL_SECONDS",
    "HEARTBEAT_STALE_SECONDS",
    "RETRYABLE_STATUSES",
    "RUN_OPS_NAME",
    "TERMINAL_STATUSES",
    "ExecutionRecord",
    "RunOpsState",
    "RunStatus",
]
