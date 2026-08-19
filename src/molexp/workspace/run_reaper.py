"""Zombie-run reaping — flip a dead-owner ``running`` run back to ``failed``.

Lives in the workspace layer (next to the run-lifecycle/ops modules) so every
verb entry point — CLI *and* server (run-recovery bug 5) — consults the same
policy before deciding what to do with a ``running`` run; it used to live in
``molexp.cli._common``, which left UI users facing a permanent 409 on runs
whose host process had died.

Policy (CLAUDE.md, "Run status x verb selection"):

* **Same host** — the recorded ``owner_pid`` is probed directly; a dead pid
  means the owner crashed and the run is reaped.
* **Cross host** (molq / SLURM workers — the normal remote scenario) — no pid
  probe is possible, so the run is reaped **only** when its ownership
  heartbeat is stale beyond :data:`~molexp.workspace.run_ops.HEARTBEAT_STALE_SECONDS`
  (refreshed every ``HEARTBEAT_INTERVAL_SECONDS`` ≈ 30 s by the owning
  worker). A fresh heartbeat, or no heartbeat at all, leaves the run alone —
  never kill a possibly-live HPC job on guesswork.
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .models import ErrorInfo, RunStatus
from .run_ops import HEARTBEAT_STALE_SECONDS

if TYPE_CHECKING:
    from .run import Run

__all__ = ["pid_alive", "reap_zombie_run"]


def pid_alive(pid: int) -> bool:
    """Return ``True`` if a process with *pid* exists on this host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_zombie_run(run: Run) -> bool:
    """Mark a stale ``RUNNING`` run as ``FAILED`` if its owner is dead.

    Reads the run's hot state from the OKF ``ops/run.json`` sidecar
    (:class:`~molexp.workspace.run_ops.RunOpsState`) per wsokf-07. Same-host
    runs are pid-probed directly: a recorded ``owner_pid`` that no longer
    exists on this host means the owner died and the run is reaped.

    Cross-host runs are reaped **only** when their ``heartbeat_at`` is stale
    beyond :data:`~molexp.workspace.run_ops.HEARTBEAT_STALE_SECONDS`. A fresh
    heartbeat, or a sidecar with no heartbeat at all, leaves the run alone.

    Returns ``True`` when the run was reaped (status flipped from
    ``running`` to ``failed``), ``False`` when the owner is (or may still
    be) alive.
    """
    state = run.read_ops()
    if state.status is not RunStatus.RUNNING:
        return False

    host = state.owner_host
    same_host = host == platform.node()
    now = datetime.now(UTC)

    if same_host:
        if state.owner_pid is not None and pid_alive(state.owner_pid):
            return False  # live owner on this host
        reason = (
            f"Run was left in 'running' state by a prior invocation "
            f"(pid={state.owner_pid or '?'} host={host or '?'}) whose process is "
            "no longer alive.  Automatically marked FAILED."
        )
    else:
        age = state.heartbeat_age(now)
        if age is None or age.total_seconds() < HEARTBEAT_STALE_SECONDS:
            # Fresh heartbeat, or no heartbeat info yet (worker still
            # starting) — assume alive.
            return False
        reason = (
            f"Run was left in 'running' state on host {host or '?'} "
            f"(pid={state.owner_pid or '?'}) and its heartbeat is "
            f"{int(age.total_seconds())}s old "
            f"(threshold {int(HEARTBEAT_STALE_SECONDS)}s).  "
            "Automatically marked FAILED."
        )

    # Status / finished / cleared-ownership are hot state → the OKF ``ops``
    # sidecar (wsokf-10). The ``error`` diagnostic stays in run.json (identity).
    naive_now = datetime.now()
    run.update_ops(
        lambda s: s.model_copy(
            update={
                "status": RunStatus.FAILED,
                "finished_at": naive_now,
                "owner_pid": None,
                "owner_host": None,
                "heartbeat_at": None,
            }
        )
    )
    run._update_metadata(
        error=ErrorInfo(
            type="ZombieRun",
            message=reason,
            timestamp=naive_now,
        ),
    )
    return True
