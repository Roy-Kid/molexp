"""``RunOpsState`` — typed model for a Run's OKF ``_ops/run.json`` sidecar.

Hot machine state (status / ownership / heartbeat / executions) lives in the
``_ops/`` operational sidecar, physically isolated from the knowledge-layer
``meta.yaml`` (the OKF identity-vs-runtime split). Reuses the existing
``workspace.models`` ``RunStatus`` / ``ExecutionRecord``; timestamps are
aware-UTC so heartbeat staleness compares correctly across hosts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from .models import ExecutionRecord, RunStatus

#: Name of the Run operational document → ``_ops/run.json``.
RUN_OPS_NAME = "run"

#: Cadence at which a live run re-stamps its heartbeat (seconds).
HEARTBEAT_INTERVAL_SECONDS = 30.0

#: A heartbeat older than this is considered stale (seconds; ~10 min).
HEARTBEAT_STALE_SECONDS = 600.0

#: Statuses from which ``resume`` / ``rerun`` apply.
RETRYABLE_STATUSES: frozenset[str] = frozenset({RunStatus.FAILED.value, RunStatus.CANCELLED.value})

#: Terminal statuses — a finished run carries a ``finished_at``.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {RunStatus.SUCCEEDED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)


def _utcnow() -> datetime:
    """Return the current time as an aware-UTC :class:`datetime`."""
    return datetime.now(UTC)


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
        """Whether ``resume`` / ``rerun`` apply (status in the retryable set)."""
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

        A never-beaten state is **not** stale (no heartbeat to age); the reaping
        policy decides what to do with such a run.
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
    "RunOpsState",
]
