"""RunMonitor: lifecycle controller for the full-screen run dashboard.

molexp owns when the dashboard opens, closes, and can be reopened.
molq owns the dashboard renderer (:class:`~molq.dashboard.RunDashboard`).

Usage (from CLI or programmatic code)::

    from molexp.monitor import RunMonitor

    monitor = RunMonitor(title="my-sweep")
    monitor.watch(runs)           # blocks until user presses 'q'
    # jobs keep running after this returns
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molexp.plugins.submit_molq.metadata import normalize_executor_info

if TYPE_CHECKING:
    from molexp.workspace.run import Run


# ── Helpers ───────────────────────────────────────────────────────────────────


def _elapsed(created_at: str | None, finished_at: str | None = None) -> str | None:
    """Compute human-readable elapsed time from ISO timestamps."""
    if not created_at:
        return None
    try:
        start = datetime.fromisoformat(created_at)
        end   = datetime.fromisoformat(finished_at) if finished_at else datetime.now()
        secs  = max(0, int((end - start).total_seconds()))
        if secs < 60:
            return f"{secs}s"
        m, s = divmod(secs, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"
    except Exception:
        return None


def _read_run_json(run_dir: Path) -> dict[str, Any]:
    """Load run.json without constructing a Run object."""
    p = run_dir / "run.json"
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return {}


def _overall_status(running: int, pending: int, failed: int, done: int) -> str:
    if running > 0:
        return "running"
    if pending > 0:
        return "pending"
    if failed > 0 and done == 0:
        return "failed"
    if failed > 0:
        return "mixed"
    return "done"


# ── RunMonitor ────────────────────────────────────────────────────────────────


class RunMonitor:
    """Lifecycle controller for the full-screen run dashboard.

    Owns when the dashboard is opened and closed.  Delegates all rendering
    to :class:`~molq.dashboard.RunDashboard` from the molq package.

    Status is refreshed by re-reading each run's ``run.json`` file on every
    tick, which works for both local and remote (cluster) runs without any
    additional IPC.

    Args:
        title: Display title shown in the monitor header.
        refresh_interval: Seconds between automatic data refreshes.
    """

    def __init__(
        self,
        title: str = "molexp",
        *,
        refresh_interval: float = 2.0,
    ) -> None:
        self._title = title
        self._refresh_interval = refresh_interval

    def watch(self, runs: list[Run]) -> None:
        """Open the full-screen dashboard and block until the user presses ``q``.

        Closing the dashboard does **not** cancel any running jobs — it only
        closes the viewer.  The caller is responsible for any post-close
        messaging (e.g. "reopen with molexp watch …").

        Args:
            runs: Run objects to monitor.  Paths are snapshot at call time;
                  status is polled from disk on every refresh.
        """
        from molq.dashboard import DashboardState, JobRow, RunDashboard

        # Snapshot (id, run_dir) pairs once — paths are stable
        run_entries: list[tuple[str, Path]] = [
            (r.id, r.run_dir) for r in runs
        ]

        def _build_state() -> DashboardState:
            rows: list[JobRow] = []
            running = pending = done = failed = 0

            for run_id, run_dir in run_entries:
                data   = _read_run_json(run_dir)
                status = data.get("status", "pending")

                elapsed = _elapsed(
                    # created_at is the best proxy we have for start time
                    data.get("created_at"),
                    data.get("finished_at"),
                )

                executor_info = normalize_executor_info(
                    data.get("executor_info"),
                    data.get("labels"),
                )
                sched_id = executor_info.get("scheduler_job_id")

                error_msg: str | None = None
                if isinstance(data.get("error"), dict):
                    error_msg = data["error"].get("message")

                profile_name = data.get("profile")
                extras: tuple[tuple[str, str], ...] = (
                    (("profile", profile_name),) if profile_name else ()
                )

                rows.append(JobRow(
                    state=status,
                    run_id=run_id,
                    cluster=executor_info.get("cluster_name"),
                    scheduler_id=sched_id,
                    elapsed=elapsed,
                    message=error_msg,
                    extras=extras,
                ))

                s = status.lower()
                if s == "running":
                    running += 1
                elif s == "pending":
                    pending += 1
                elif s in ("succeeded", "done"):
                    done += 1
                elif s in ("failed", "cancelled"):
                    failed += 1
                else:
                    pending += 1  # unknown → treat as pending

            return DashboardState(
                title=self._title,
                overall_status=_overall_status(running, pending, failed, done),
                total=len(run_entries),
                running=running,
                pending=pending,
                done=done,
                failed=failed,
                updated_at=datetime.now().strftime("%H:%M:%S"),
                jobs=tuple(rows),
            )

        RunDashboard().watch(_build_state, refresh_interval=self._refresh_interval)
