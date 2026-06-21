"""Run lifecycle on the knowledge layer (okf-05-02).

Ownership claim, the heartbeat daemon, terminal finish, and the stale-running
reaping decision — all driving okf-04 :class:`RunOpsState` on a knowledge
:class:`Run`. Mirrors the semantics of ``molexp.workspace.run_lifecycle`` +
``molexp.cli._common.reap_zombie_run`` (CLAUDE.md's run-status/verb law) but on
the typed ``_ops/run.json`` instead of string labels. The reaping rule is a
**pure predicate** (:func:`should_reap`) plus a thin OS-probing caller
(:func:`reap_run_if_stale`).
"""

from __future__ import annotations

import os
import threading
from datetime import datetime

from mollog import get_logger

from .concepts import Run
from .ops import HEARTBEAT_INTERVAL_SECONDS, RunOpsState, RunStatus, _utcnow

_logger = get_logger(__name__)


def claim_ownership(
    run: Run,
    *,
    pid: int,
    host: str,
    execution_id: str | None = None,
    now: datetime | None = None,
) -> RunOpsState:
    """Mark *run* RUNNING and stamp this process's ownership.

    Sets ``owner_pid`` / ``owner_host`` / ``heartbeat_at`` and (on the first
    claim) ``started_at``; clears ``finished_at``; records ``current_execution_id``
    when given. Idempotent on ``started_at`` (re-claim preserves it).
    """
    ts = now or _utcnow()

    def claim(state: RunOpsState) -> RunOpsState:
        update: dict[str, object] = {
            "status": RunStatus.RUNNING,
            "owner_pid": pid,
            "owner_host": host,
            "heartbeat_at": ts,
            "finished_at": None,
        }
        if state.started_at is None:
            update["started_at"] = ts
        if execution_id is not None:
            update["current_execution_id"] = execution_id
        return state.model_copy(update=update)

    return run.update_ops(claim)


def finish_run(run: Run, status: RunStatus, *, now: datetime | None = None) -> RunOpsState:
    """Terminally finish *run*: set *status* + ``finished_at``, clear ownership.

    Also closes the open current :class:`ExecutionRecord` (stamps its
    ``finished_at`` + ``status``). Records state; does not police legality.
    """
    ts = now or _utcnow()

    def finish(state: RunOpsState) -> RunOpsState:
        update: dict[str, object] = {
            "status": status,
            "finished_at": ts,
            "owner_pid": None,
            "owner_host": None,
            "heartbeat_at": None,
        }
        if state.current_execution_id is not None:
            update["executions"] = tuple(
                record.model_copy(update={"finished_at": ts, "status": status.value})
                if record.execution_id == state.current_execution_id and record.finished_at is None
                else record
                for record in state.executions
            )
        return state.model_copy(update=update)

    return run.update_ops(finish)


def should_reap(
    state: RunOpsState,
    *,
    now: datetime,
    current_host: str,
    pid_alive: bool | None,
) -> bool:
    """Whether a stale RUNNING run should be reaped to FAILED (pure decision).

    Only RUNNING runs are candidates. **Same host**: reap iff the recorded
    owner pid is dead (``pid_alive is False``). **Cross host**: reap iff the
    heartbeat is stale beyond ``HEARTBEAT_STALE_SECONDS``; a fresh or absent
    heartbeat is never reaped (a live HPC job).
    """
    if state.status != RunStatus.RUNNING:
        return False
    if state.owner_host == current_host:
        return pid_alive is False
    return state.is_heartbeat_stale(now)


def _pid_alive(pid: int) -> bool:
    """Whether *pid* is alive on this host (``os.kill(pid, 0)`` probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return True


def reap_run_if_stale(run: Run, *, current_host: str, now: datetime | None = None) -> bool:
    """Reap *run* to FAILED if its owner is dead/stale; return whether reaped."""
    ts = now or _utcnow()
    state = run.read_ops()
    if state.owner_host == current_host and state.owner_pid is not None:
        pid_alive: bool | None = _pid_alive(state.owner_pid)
    else:
        pid_alive = None
    if should_reap(state, now=ts, current_host=current_host, pid_alive=pid_alive):
        finish_run(run, RunStatus.FAILED, now=ts)
        return True
    return False


class RunHeartbeat:
    """Daemon thread that re-stamps *run*'s heartbeat until stopped.

    A plain runtime container (not pydantic). Use via ``start()`` / ``stop()``
    or as a context manager. Stop it **before** the terminal write so the
    reaper never sees a fresh beat on a finished run.
    """

    def __init__(self, run: Run, *, interval: float = HEARTBEAT_INTERVAL_SECONDS) -> None:
        self._run = run
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the heartbeat daemon (no-op if already running)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"molexp-knowledge-heartbeat-{self._run.name}",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._run.beat()
            except Exception:
                # A transient write hiccup must never kill the daemon — a
                # missed beat only delays staleness detection, whereas a dead
                # thread would let the reaper kill a live run.
                _logger.debug(f"heartbeat beat failed for run {self._run.name}", exc_info=True)

    def stop(self) -> None:
        """Signal the daemon to exit and join it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_alive(self) -> bool:
        """Whether the heartbeat daemon thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> RunHeartbeat:
        self.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False


__all__ = [
    "RunHeartbeat",
    "claim_ownership",
    "finish_run",
    "reap_run_if_stale",
    "should_reap",
]
