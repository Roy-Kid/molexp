"""Shared helpers for the molexp CLI.

Everything in this module is internal — command modules import from it.
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from pathlib import Path

from rich import print as rprint
from rich.console import Console

from molexp._typing import JSONValue
from molexp.plugins.submit_molq.metadata import normalize_executor_info
from molexp.workspace import Workspace
from molexp.workspace.run import Run

console = Console()

# Terminal run statuses — not cancellable / considered "done".
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

# Rich color mapping used by list / info / monitor displays.
_STATUS_COLORS: dict[str, str] = {
    "succeeded": "green",
    "failed": "red",
    "running": "yellow",
    "pending": "blue",
    "cancelled": "gray",
}


def status_color(status: str) -> str:
    """Return the rich color for a run status (white if unknown)."""
    return _STATUS_COLORS.get(str(status).lower(), "white")


def get_workspace(path: Path | None = None) -> Workspace:
    """Load the workspace at *path* (default: current directory)."""
    return Workspace(path or Path.cwd())


def deterministic_run_id(params: dict[str, JSONValue]) -> str:
    """Generate a deterministic 16-char run ID from parameters.

    Same parameters always produce the same ID, making run creation
    idempotent across repeated ``molexp run`` invocations.  The caller
    decides which fields to include (for profile-aware IDs, mix in
    the profile name / config hash).

    Delegates to :func:`molexp.workspace.utils.derive_run_id` — the single
    canonicalization shared with ``Experiment.add_runs`` — keeping this name
    and its 16-char output stable for existing CLI callers.
    """
    from molexp.workspace.utils import derive_run_id

    return derive_run_id(params)


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


# A cross-host ``running`` run is only considered a zombie when its
# ownership heartbeat (refreshed every
# ``molexp.workspace.run_lifecycle.HEARTBEAT_INTERVAL_SECONDS`` ≈ 30 s by
# the owning worker) is stale beyond this generous threshold. Cross-host
# execution via molq / SLURM is the product's core scenario — when in
# doubt, leave the run alone rather than kill a live HPC job.
CROSS_HOST_HEARTBEAT_STALE_SECONDS = 600.0  # 10 minutes


def reap_zombie_run(run: Run) -> bool:
    """Mark a stale ``RUNNING`` run as ``FAILED`` if its owner is dead.

    Reads the run's hot state from the OKF ``_ops/run.json`` sidecar
    (:class:`RunOpsState`) per wsokf-07. Same-host runs are pid-probed
    directly: a recorded ``owner_pid`` that no longer exists on this host
    means the owner died and the run is reaped.

    Cross-host runs (molq / SLURM workers — the normal remote scenario)
    cannot be pid-probed from here, so they are reaped **only** when their
    ``heartbeat_at`` is stale beyond :data:`CROSS_HOST_HEARTBEAT_STALE_SECONDS`
    (which equals ``run_ops.HEARTBEAT_STALE_SECONDS``). A fresh heartbeat, or a
    sidecar with no heartbeat at all, leaves the run alone — never kill a
    possibly-live HPC run on guesswork.

    Returns ``True`` when the run was reaped (status flipped from
    ``running`` to ``failed``), ``False`` when the owner is (or may still
    be) alive.
    """
    from molexp.workspace.models import ErrorInfo, RunStatus

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
        if age is None or age.total_seconds() < CROSS_HOST_HEARTBEAT_STALE_SECONDS:
            # Fresh heartbeat, or no heartbeat info yet (worker still
            # starting) — assume alive.
            return False
        reason = (
            f"Run was left in 'running' state on host {host or '?'} "
            f"(pid={state.owner_pid or '?'}) and its heartbeat is "
            f"{int(age.total_seconds())}s old "
            f"(threshold {int(CROSS_HOST_HEARTBEAT_STALE_SECONDS)}s).  "
            "Automatically marked FAILED."
        )

    # Status / finished / cleared-ownership are hot state → the OKF ``_ops``
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


def run_executor_info(run: Run) -> dict[str, str]:
    """Return normalized executor metadata for a workspace run.

    Ownership (pid/host) now lives in the OKF ``_ops`` sidecar (wsokf-10); it
    is surfaced as label fallbacks for ``normalize_executor_info``, which only
    consults scheduler-shaped keys (never pid/host), so an empty labels map is
    sufficient here.
    """
    return normalize_executor_info(run.metadata.executor_info, {})


__all__ = [
    "_STATUS_COLORS",
    "_TERMINAL_STATUSES",
    "console",
    "deterministic_run_id",
    "get_workspace",
    "pid_alive",
    "reap_zombie_run",
    "rprint",
    "run_executor_info",
    "status_color",
]
