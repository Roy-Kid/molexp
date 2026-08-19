"""Heartbeat refresh on running runs (``RunLifecycle.refresh_heartbeat``).

The ownership stamp (``owner_pid`` / ``owner_host`` / ``heartbeat_at`` on the
OKF ``ops/run.json`` sidecar) is written once at claim time; a background
daemon thread keeps ``heartbeat_at`` fresh while the run executes so cross-host
reapers can tell a live remote run from a zombie.
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_ops(run) -> dict:
    return json.loads(Path(str(run.run_dir / "ops" / "run.json")).read_text())


class TestRefreshHeartbeat:
    def test_refresh_updates_only_the_heartbeat(self, run) -> None:
        ctx = run.start()
        with ctx:
            before = _read_ops(run)
            ctx._lifecycle.refresh_heartbeat()
            after = _read_ops(run)

            assert after["heartbeat_at"] >= before["heartbeat_at"]
            # Ownership + status preserved verbatim.
            assert after["owner_pid"] == before["owner_pid"]
            assert after["owner_host"] == before["owner_host"]
            assert after["status"] == before["status"]

    def test_refresh_is_noop_before_first_ops_write(self, run, experiment) -> None:
        # A run whose ops/run.json does not exist yet (no ownership claim)
        # must not be resurrected by a stray heartbeat tick.
        fresh = experiment.add_run(params={"lr": 9e-9})
        ctx = fresh.start()
        ops_json = Path(str(fresh.run_dir / "ops" / "run.json"))
        if ops_json.exists():
            ops_json.unlink()
        ctx._lifecycle.refresh_heartbeat()
        assert not ops_json.exists()
