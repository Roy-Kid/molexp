"""Heartbeat refresh on running runs (``RunLifecycle``).

The ownership stamp (``pid`` / ``host`` / ``heartbeat`` labels) is written
once at claim time; a background daemon thread must keep ``heartbeat``
fresh while the run executes so cross-host reapers can tell a live remote
run from a zombie. On exit the stamp is removed entirely.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from molexp.workspace.run_lifecycle import HEARTBEAT_INTERVAL_SECONDS


def _read_run_json(run) -> dict:
    return json.loads(Path(str(run.run_dir / "run.json")).read_text())


class TestHeartbeatStamp:
    def test_claim_stamps_heartbeat_label(self, run) -> None:
        with run.start():
            labels = _read_run_json(run)["labels"]
            assert "pid" in labels
            assert "host" in labels
            # The stamp must parse as an ISO timestamp.
            datetime.fromisoformat(labels["heartbeat"])

    def test_exit_removes_ownership_stamp(self, run) -> None:
        with run.start():
            pass
        labels = _read_run_json(run)["labels"]
        assert "pid" not in labels
        assert "host" not in labels
        assert "heartbeat" not in labels

    def test_default_interval_is_seconds_scale(self) -> None:
        # Guard against accidental unit drift (minutes / ms) — the reaper
        # staleness threshold (600 s) assumes a tens-of-seconds cadence.
        assert 5.0 <= HEARTBEAT_INTERVAL_SECONDS <= 120.0


class TestRefreshHeartbeat:
    def test_refresh_updates_only_the_heartbeat_label(self, run) -> None:
        ctx = run.start()
        with ctx:
            before = _read_run_json(run)
            ctx._lifecycle.refresh_heartbeat()
            after = _read_run_json(run)

            assert after["labels"]["heartbeat"] >= before["labels"]["heartbeat"]
            # Everything else is preserved verbatim — including the
            # ``context`` blob ContextStore wrote, which a naive
            # metadata-only rewrite would drop.
            assert after["labels"]["pid"] == before["labels"]["pid"]
            assert after["labels"]["host"] == before["labels"]["host"]
            assert after["status"] == before["status"]
            assert after.get("context") == before.get("context")
            assert "context" in after

    def test_refresh_updates_in_memory_labels(self, run) -> None:
        ctx = run.start()
        with ctx:
            ctx._lifecycle.refresh_heartbeat()
            on_disk = _read_run_json(run)["labels"]["heartbeat"]
            assert run.metadata.labels["heartbeat"] == on_disk

    def test_refresh_is_noop_before_first_write(self, run, experiment) -> None:
        # A run whose run.json does not exist yet must not be created by a
        # stray heartbeat tick.
        fresh = experiment.add_run(params={"lr": 9e-9})
        ctx = fresh.start()
        # Do not enter — exercise refresh directly against a missing file.
        run_json = Path(str(fresh.run_dir / "run.json"))
        if run_json.exists():
            run_json.unlink()
        ctx._lifecycle.refresh_heartbeat()
        assert not run_json.exists()


class TestHeartbeatThread:
    def test_heartbeat_refreshes_periodically_while_running(self, run) -> None:
        ctx = run.start()
        ctx._lifecycle._heartbeat_interval = 0.05
        with ctx:
            first = _read_run_json(run)["labels"]["heartbeat"]
            deadline = time.monotonic() + 3.0
            current = first
            while current == first and time.monotonic() < deadline:
                time.sleep(0.02)
                current = _read_run_json(run)["labels"]["heartbeat"]
            assert current > first, "heartbeat label was never refreshed"

    def test_heartbeat_thread_stops_on_exit(self, run) -> None:
        ctx = run.start()
        ctx._lifecycle._heartbeat_interval = 0.05
        with ctx:
            pass
        assert ctx._lifecycle._heartbeat_thread is None
        # No revival: the label stays absent after exit.
        time.sleep(0.15)
        assert "heartbeat" not in _read_run_json(run)["labels"]
