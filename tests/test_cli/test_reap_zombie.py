"""Zombie-run reaping — the workspace ``run_reaper.reap_zombie_run`` rules.

Sole owner of the reaper's own state transitions (``tests/test_workspace/
test_lifecycle_ops.py`` explicitly defers the dead-pid / cross-host-heartbeat
rules here and does not re-test them). Reaper state lives in the typed OKF
``_ops/run.json`` sidecar (:class:`RunOpsState`): same-host runs are pid-probed
directly; cross-host runs (molq / SLURM) are reaped only when ``heartbeat_at``
is stale beyond ``HEARTBEAT_STALE_SECONDS`` — a fresh or absent heartbeat leaves
them alone.
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime, timedelta

import pytest

from molexp.cli._common import (
    CROSS_HOST_HEARTBEAT_STALE_SECONDS,
    reap_zombie_run,
)
from molexp.workspace import Workspace
from molexp.workspace.run_ops import RunOpsState


@pytest.fixture
def running_run(tmp_path):
    ws = Workspace(root=tmp_path, name="lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
    run = exp.add_run(params={"seed": 1})
    run.materialize()
    return run


def _mark_running(
    run,
    *,
    pid: int,
    host: str,
    heartbeat_at: datetime | None,
) -> None:
    run.write_ops(
        RunOpsState(
            status="running",
            owner_pid=pid,
            owner_host=host,
            heartbeat_at=heartbeat_at,
        )
    )


def _dead_pid() -> int:
    """A pid that is exceedingly unlikely to exist."""
    pid = 2**22 - 7
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    return pid


class TestReapZombieRun:
    """State transitions of ``reap_zombie_run`` (one per reaping rule)."""

    def test_same_host_live_pid_is_left_alone(self, running_run) -> None:
        _mark_running(
            running_run,
            pid=os.getpid(),
            host=platform.node(),
            heartbeat_at=datetime.now(UTC),
        )
        assert reap_zombie_run(running_run) is False
        assert running_run.status == "running"

    def test_same_host_dead_pid_is_reaped_and_ownership_cleared(self, running_run) -> None:
        _mark_running(
            running_run,
            pid=_dead_pid(),
            host=platform.node(),
            heartbeat_at=datetime.now(UTC),
        )
        assert reap_zombie_run(running_run) is True
        assert running_run.status == "failed"
        state = running_run.read_ops()
        assert state.owner_pid is None
        assert state.heartbeat_at is None

    def test_cross_host_fresh_heartbeat_is_left_alone(self, running_run) -> None:
        _mark_running(
            running_run,
            pid=12345,
            host="hpc-login-01",
            heartbeat_at=datetime.now(UTC),
        )
        assert reap_zombie_run(running_run) is False
        assert running_run.status == "running"

    def test_cross_host_missing_heartbeat_is_left_alone(self, running_run) -> None:
        _mark_running(running_run, pid=12345, host="hpc-login-01", heartbeat_at=None)
        assert reap_zombie_run(running_run) is False
        assert running_run.status == "running"

    def test_cross_host_stale_heartbeat_is_reaped(self, running_run) -> None:
        stale = datetime.now(UTC) - timedelta(seconds=CROSS_HOST_HEARTBEAT_STALE_SECONDS + 60)
        _mark_running(running_run, pid=12345, host="hpc-login-01", heartbeat_at=stale)
        assert reap_zombie_run(running_run) is True
        assert running_run.status == "failed"
