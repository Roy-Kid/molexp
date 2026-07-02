"""Shared helpers for the molexp CLI.

Everything in this module is internal — command modules import from it.
"""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint
from rich.console import Console

from molexp._typing import JSONValue
from molexp.plugins.submit_molq.metadata import normalize_executor_info
from molexp.workspace import Workspace

# Zombie-run reaping moved into the workspace layer (run-recovery bug 5) so
# the CLI and the server verbs consult ONE policy; these re-exports keep the
# historical ``molexp.cli._common`` import path working.
from molexp.workspace.run import Run
from molexp.workspace.run_ops import HEARTBEAT_STALE_SECONDS
from molexp.workspace.run_reaper import pid_alive, reap_zombie_run

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


# Backward-compatible alias — the canonical constant lives in
# ``molexp.workspace.run_ops.HEARTBEAT_STALE_SECONDS`` (one source of truth
# for the cross-host staleness threshold).
CROSS_HOST_HEARTBEAT_STALE_SECONDS = HEARTBEAT_STALE_SECONDS


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
