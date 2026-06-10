"""Shared helpers for the molexp CLI.

Everything in this module is internal — command modules import from it.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime
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


def _heartbeat_age_seconds(labels: dict[str, str], now: datetime) -> float | None:
    """Age of the ``heartbeat`` ownership label, or ``None`` when absent/unparseable."""
    raw = labels.get("heartbeat")
    if not raw:
        return None
    try:
        return (now - datetime.fromisoformat(raw)).total_seconds()
    except ValueError:
        return None


def reap_zombie_run(run: Run) -> bool:
    """Mark a stale ``RUNNING`` run as ``FAILED`` if its owner is dead.

    Same-host runs are probed directly: a recorded pid that no longer
    exists on this host means the owner died and the run is reaped.

    Cross-host runs (molq / SLURM workers — the normal remote scenario)
    cannot be pid-probed from here, so they are reaped **only** when their
    ownership heartbeat is stale beyond
    :data:`CROSS_HOST_HEARTBEAT_STALE_SECONDS`. A fresh heartbeat, or a
    run.json predating the heartbeat label entirely, leaves the run alone
    — never kill a possibly-live HPC run on guesswork.

    Returns ``True`` when the run was reaped (status flipped from
    ``running`` to ``failed``), ``False`` when the owner is (or may still
    be) alive.
    """
    from molexp.workspace.models import ErrorInfo
    from molexp.workspace.run import RunStatus

    labels = dict(run.metadata.labels)
    pid_str = labels.get("pid")
    host = labels.get("host")
    same_host = host == platform.node()
    now = datetime.now()

    if same_host:
        if pid_str and pid_str.isdigit() and pid_alive(int(pid_str)):
            return False  # live owner on this host
        reason = (
            f"Run was left in 'running' state by a prior invocation "
            f"(pid={pid_str or '?'} host={host or '?'}) whose process is "
            "no longer alive.  Automatically marked FAILED."
        )
    else:
        age = _heartbeat_age_seconds(labels, now)
        if age is None or age < CROSS_HOST_HEARTBEAT_STALE_SECONDS:
            # Fresh heartbeat, or no heartbeat info yet (pre-heartbeat
            # run.json / worker still starting) — assume alive.
            return False
        reason = (
            f"Run was left in 'running' state on host {host or '?'} "
            f"(pid={pid_str or '?'}) and its heartbeat is {int(age)}s old "
            f"(threshold {int(CROSS_HOST_HEARTBEAT_STALE_SECONDS)}s).  "
            "Automatically marked FAILED."
        )

    for key in ("pid", "host", "heartbeat"):
        labels.pop(key, None)
    run._update_metadata(
        status=RunStatus.FAILED,
        finished_at=now,
        labels=labels,
        error=ErrorInfo(
            type="ZombieRun",
            message=reason,
            timestamp=now,
        ),
    )
    return True


def run_executor_info(run: Run) -> dict[str, str]:
    """Return normalized executor metadata for a workspace run."""
    return normalize_executor_info(run.metadata.executor_info, run.metadata.labels)


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
