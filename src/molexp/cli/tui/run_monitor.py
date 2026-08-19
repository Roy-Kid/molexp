"""RunMonitor: lifecycle controller for the full-screen run dashboard.

molexp owns when the dashboard opens, closes, and can be reopened.
molq owns the dashboard renderer (:class:`~molq.dashboard.RunDashboard`).

Usage (from CLI or programmatic code)::

    from molexp.cli.tui import RunMonitor

    monitor = RunMonitor(title="my-experiment")
    monitor.watch(runs)  # blocks until user presses 'q'
    # jobs keep running after this returns
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from molexp._run_display import elapsed as _elapsed
from molexp.plugins.submit_molq.metadata import normalize_executor_info

if TYPE_CHECKING:
    from molexp.workspace.run import Run


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _run_label(r: Run) -> str | None:
    """Human-meaningful row label (experiment name + optional replica)."""
    exp_name = r.experiment.name
    replica = r.metadata.parameters.get("replica")
    if replica is not None and exp_name:
        return f"{exp_name}#{replica}"
    return exp_name or None


# ── RunMonitor ────────────────────────────────────────────────────────────────


class RunMonitor:
    """Lifecycle controller for the full-screen run dashboard.

    Owns when the dashboard is opened and closed.  Delegates all rendering
    to :class:`~molq.dashboard.RunDashboard` from the molq package.

    Status is refreshed by re-reading each :class:`~molexp.workspace.run.Run`
    through its FileSystem on every tick — works for local *and* remote
    workspaces (no local ``Path(run_dir)`` open).

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
            runs: Run objects to monitor.  Status is polled via each Run's
                  FileSystem on every refresh (local or remote).
        """
        from molq.dashboard import DashboardState, JobRow, RunDashboard

        # Keep live Run handles so each tick re-reads ops through ``fs``.
        run_list = list(runs)

        def _build_state() -> DashboardState:
            rows: list[JobRow] = []
            running = pending = done = failed = 0

            for r in run_list:
                run_id = r.id
                run_name = _run_label(r)
                try:
                    status = str(r.status)
                except Exception:
                    status = "pending"

                created_at = r.metadata.created_at.isoformat() if r.metadata.created_at else None
                finished = r.finished_at
                finished_at = finished.isoformat() if finished is not None else None
                elapsed = _elapsed(created_at, finished_at)

                labels_raw = getattr(r.metadata, "labels", None)
                executor_info = normalize_executor_info(
                    r.metadata.executor_info
                    if isinstance(r.metadata.executor_info, dict)
                    else None,
                    (
                        {str(k): v for k, v in labels_raw.items() if isinstance(v, str)}
                        if isinstance(labels_raw, dict)
                        else None
                    ),
                )
                sched_id = executor_info.get("scheduler_job_id")

                error_msg: str | None = None
                err = r.metadata.error
                if err is not None:
                    error_msg = getattr(err, "message", None) or str(err)

                profile_name = r.metadata.profile or None
                extras: tuple[tuple[str, str], ...] = (
                    (("profile", profile_name),) if profile_name else ()
                )

                # ``run_name`` is folded into ``extras`` because molq's
                # ``JobRow`` has no dedicated name field — the dashboard
                # already groups by ``run_id`` and surfaces ``extras`` in
                # the row's secondary line.
                if run_name and run_name != run_id:
                    extras = (*extras, ("name", run_name))
                rows.append(
                    JobRow(
                        state=status,
                        run_id=run_id,
                        cluster=executor_info.get("cluster_name"),
                        scheduler_id=sched_id,
                        elapsed=elapsed,
                        message=error_msg,
                        extras=extras,
                    )
                )

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
                total=len(run_list),
                running=running,
                pending=pending,
                done=done,
                failed=failed,
                updated_at=datetime.now().strftime("%H:%M:%S"),
                jobs=tuple(rows),
            )

        RunDashboard().watch(_build_state, refresh_interval=self._refresh_interval)
