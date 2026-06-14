"""Zombie-run reaping (``molexp.cli._common.reap_zombie_run``).

Same-host runs are pid-probed directly. Cross-host runs (molq / SLURM —
the product's core remote scenario) must NEVER be reaped while their
heartbeat is fresh or while no heartbeat information exists; only a
heartbeat stale beyond ``CROSS_HOST_HEARTBEAT_STALE_SECONDS`` flips them
to failed.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime, timedelta

import pytest

from molexp.cli._common import (
    CROSS_HOST_HEARTBEAT_STALE_SECONDS,
    reap_zombie_run,
)
from molexp.workspace import Workspace
from molexp.workspace.run_lifecycle import HEARTBEAT_INTERVAL_SECONDS


@pytest.fixture
def running_run(tmp_path):
    ws = Workspace(root=tmp_path, name="lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
    run = exp.add_run(params={"seed": 1})
    run.materialize()
    return run


def _mark_running(run, labels: dict[str, str]) -> None:
    run._update_metadata(status="running", labels=labels)


def _dead_pid() -> int:
    """A pid that is exceedingly unlikely to exist."""
    pid = 2**22 - 7
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    return pid


class TestSameHost:
    def test_live_pid_is_left_alone(self, running_run) -> None:
        _mark_running(
            running_run,
            {
                "pid": str(os.getpid()),
                "host": platform.node(),
                "heartbeat": datetime.now().isoformat(),
            },
        )
        assert reap_zombie_run(running_run) is False
        assert running_run.metadata.status == "running"

    def test_dead_pid_is_reaped(self, running_run) -> None:
        _mark_running(
            running_run,
            {
                "pid": str(_dead_pid()),
                "host": platform.node(),
                "heartbeat": datetime.now().isoformat(),
            },
        )
        assert reap_zombie_run(running_run) is True
        assert running_run.metadata.status == "failed"
        assert running_run.metadata.error is not None
        assert running_run.metadata.error.type == "ZombieRun"
        # Ownership stamp is removed on reap.
        for key in ("pid", "host", "heartbeat"):
            assert key not in running_run.metadata.labels


class TestCrossHost:
    """Cross-host = recorded host differs from platform.node()."""

    def test_fresh_heartbeat_is_left_alone(self, running_run) -> None:
        _mark_running(
            running_run,
            {
                "pid": "12345",
                "host": "hpc-login-01",
                "heartbeat": datetime.now().isoformat(),
            },
        )
        assert reap_zombie_run(running_run) is False
        assert running_run.metadata.status == "running"

    def test_missing_heartbeat_is_left_alone(self, running_run) -> None:
        # Backward compatibility: run.json written before the heartbeat
        # label existed must never be reaped cross-host.
        _mark_running(running_run, {"pid": "12345", "host": "hpc-login-01"})
        assert reap_zombie_run(running_run) is False
        assert running_run.metadata.status == "running"

    def test_unparseable_heartbeat_is_left_alone(self, running_run) -> None:
        _mark_running(
            running_run,
            {"pid": "12345", "host": "hpc-login-01", "heartbeat": "not-a-timestamp"},
        )
        assert reap_zombie_run(running_run) is False
        assert running_run.metadata.status == "running"

    def test_stale_heartbeat_is_reaped(self, running_run) -> None:
        stale = datetime.now() - timedelta(seconds=CROSS_HOST_HEARTBEAT_STALE_SECONDS + 60)
        _mark_running(
            running_run,
            {"pid": "12345", "host": "hpc-login-01", "heartbeat": stale.isoformat()},
        )
        assert reap_zombie_run(running_run) is True
        assert running_run.metadata.status == "failed"
        assert running_run.metadata.error is not None
        assert "heartbeat" in running_run.metadata.error.message

    def test_just_under_threshold_is_left_alone(self, running_run) -> None:
        recent = datetime.now() - timedelta(seconds=CROSS_HOST_HEARTBEAT_STALE_SECONDS - 60)
        _mark_running(
            running_run,
            {"pid": "12345", "host": "hpc-login-01", "heartbeat": recent.isoformat()},
        )
        assert reap_zombie_run(running_run) is False


class TestThresholds:
    def test_staleness_threshold_dominates_refresh_cadence(self) -> None:
        # The reaper must tolerate many missed beats before declaring a
        # cross-host run dead.
        assert CROSS_HOST_HEARTBEAT_STALE_SECONDS >= 10 * HEARTBEAT_INTERVAL_SECONDS
