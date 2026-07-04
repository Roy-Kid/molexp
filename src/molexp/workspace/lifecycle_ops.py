"""Run-lifecycle operation cores shared by CLI, server, and harness capabilities.

Workspace-layer homes for verb bodies that must be reachable from the harness
capability handlers (harness→workspace is legal; harness→services is not).
Each op enforces the frozen verb law itself — the capability layer adds
gating, never semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run import Run

__all__ = ["cancel_run"]


def cancel_run(run: Run) -> None:
    """Cancel an actively-running *run* — the canonical intervene verb.

    Reaps first (every verb entry reaps — a stale ``running`` with a dead
    owner flips to ``failed`` before the verb decides), then refuses any
    non-``running`` status loudly: there is nothing to intervene on.

    Args:
        run: The run to cancel.

    Raises:
        ValueError: The run is not ``running`` after reaping. The message
            names the actual status and the verb that owns it (``resume`` /
            ``rerun`` for failed/cancelled; nothing for succeeded; plain
            ``run`` for pending).
    """
    from .run_reaper import reap_zombie_run

    reap_zombie_run(run)
    status = str(run.status).lower()
    if status != "running":
        owner = {
            "pending": "it has not started — `molexp run` owns pending runs",
            "failed": "it already stopped — resume/rerun own failed runs",
            "cancelled": "it is already cancelled — resume/rerun own cancelled runs",
            "succeeded": "it finished — read its results instead",
        }.get(status, "only a running run can be cancelled")
        raise ValueError(f"run {run.id} is {status!r}, not running; {owner}")
    run.cancel()
