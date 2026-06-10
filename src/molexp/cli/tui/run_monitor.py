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
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from molexp._run_display import elapsed as _elapsed
from molexp._run_display import read_run_json as _read_run_json
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

        # Snapshot (id, name, run_dir) triples once — paths are stable.
        # ``name`` is the experiment name (the human-meaningful label); when
        # an experiment has multiple replicas we append ``#<replica>`` from
        # run.parameters so rows stay distinguishable.
        run_entries: list[tuple[str, str | None, Path | PurePath]] = []
        for r in runs:
            exp_name = r.experiment.name
            replica = r.metadata.parameters.get("replica")
            label: str | None = exp_name
            if replica is not None and exp_name:
                label = f"{exp_name}#{replica}"
            run_entries.append((r.id, label, r.run_dir))

        def _build_state() -> DashboardState:
            rows: list[JobRow] = []
            running = pending = done = failed = 0

            for run_id, run_name, run_dir in run_entries:
                data = _read_run_json(run_dir)
                status_raw = data.get("status")
                status = status_raw if isinstance(status_raw, str) else "pending"

                created_at = data.get("created_at")
                finished_at = data.get("finished_at")
                elapsed = _elapsed(
                    # created_at is the best proxy we have for start time
                    created_at if isinstance(created_at, str) else None,
                    finished_at if isinstance(finished_at, str) else None,
                )

                info_raw = data.get("executor_info")
                labels_raw = data.get("labels")
                executor_info = normalize_executor_info(
                    info_raw if isinstance(info_raw, dict) else None,
                    (
                        {str(k): v for k, v in labels_raw.items() if isinstance(v, str)}
                        if isinstance(labels_raw, dict)
                        else None
                    ),
                )
                sched_id = executor_info.get("scheduler_job_id")

                error_msg: str | None = None
                err = data.get("error")
                if isinstance(err, dict):
                    msg_raw = err.get("message")
                    error_msg = msg_raw if isinstance(msg_raw, str) else None

                profile_raw = data.get("profile")
                profile_name = profile_raw if isinstance(profile_raw, str) else None
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
                total=len(run_entries),
                running=running,
                pending=pending,
                done=done,
                failed=failed,
                updated_at=datetime.now().strftime("%H:%M:%S"),
                jobs=tuple(rows),
            )

        RunDashboard().watch(_build_state, refresh_interval=self._refresh_interval)
