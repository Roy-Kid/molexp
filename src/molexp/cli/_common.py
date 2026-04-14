"""Shared helpers for the molexp CLI.

Everything in this module is internal — command modules import from it.
"""

from __future__ import annotations

import hashlib
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import print as rprint
from rich.console import Console

from molexp.plugins.submit_molq.metadata import normalize_executor_info
from molexp.workspace import Workspace

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


def deterministic_run_id(params: dict[str, Any]) -> str:
    """Generate a deterministic 16-char run ID from parameters.

    Same parameters always produce the same ID, making run creation
    idempotent across repeated ``molexp run`` invocations.  The caller
    decides which fields to include (for profile-aware IDs, mix in
    the profile name / config hash).
    """
    raw = "|".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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


def reap_zombie_run(run: Any) -> bool:
    """Mark a stale ``RUNNING`` run as ``FAILED`` if its owner is dead.

    Returns ``True`` when the run was reaped (status flipped from
    ``running`` to ``failed``), ``False`` when a live owner is detected.
    """
    from molexp.workspace.models import ErrorInfo
    from molexp.workspace.run import RunStatus

    labels = dict(run.metadata.labels)
    pid_str = labels.get("pid")
    host = labels.get("host")
    same_host = host == platform.node()

    if same_host and pid_str and pid_str.isdigit() and pid_alive(int(pid_str)):
        return False

    now = datetime.now()
    for key in ("pid", "host", "heartbeat"):
        labels.pop(key, None)
    run._update_metadata(
        status=RunStatus.FAILED,
        finished_at=now,
        labels=labels,
        error=ErrorInfo(
            type="ZombieRun",
            message=(
                f"Run was left in 'running' state by a prior invocation "
                f"(pid={pid_str or '?'} host={host or '?'}) that did not "
                "finish cleanly.  Automatically marked FAILED."
            ),
            timestamp=now,
        ),
    )
    return True


def run_executor_info(run: Any) -> dict[str, str]:
    """Return normalized executor metadata for a workspace run."""
    return normalize_executor_info(run.metadata.executor_info, run.metadata.labels)


__all__ = [
    "console",
    "rprint",
    "_TERMINAL_STATUSES",
    "_STATUS_COLORS",
    "status_color",
    "get_workspace",
    "deterministic_run_id",
    "pid_alive",
    "reap_zombie_run",
    "run_executor_info",
]
