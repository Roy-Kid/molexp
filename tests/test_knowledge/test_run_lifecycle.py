"""Tests for the run lifecycle primitives (okf-05-02).

Ownership claim, heartbeat daemon, terminal finish, and the stale-running
reaping decision — all driving okf-04 ``RunOpsState`` on a knowledge ``Run``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import molexp.knowledge.run_lifecycle as rl
from molexp.knowledge import (
    ExecutionRecord,
    Run,
    RunHeartbeat,
    RunOpsState,
    RunStatus,
    claim_ownership,
    finish_run,
    reap_run_if_stale,
    should_reap,
)
from molexp.knowledge.ops import HEARTBEAT_STALE_SECONDS

FIXED = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


# ── claim_ownership / finish_run (ac-002 / ac-003) ───────────────────────────


def test_claim_ownership_sets_running_and_timestamps(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    s = claim_ownership(run, pid=111, host="h1", execution_id="exec-r1-1", now=FIXED)
    assert s.status == RunStatus.RUNNING
    assert s.owner_pid == 111
    assert s.owner_host == "h1"
    assert s.heartbeat_at == FIXED
    assert s.started_at == FIXED
    assert s.finished_at is None
    assert s.current_execution_id == "exec-r1-1"

    later = FIXED + timedelta(minutes=1)
    s2 = claim_ownership(run, pid=111, host="h1", now=later)
    assert s2.started_at == FIXED  # preserved across re-claim
    assert s2.heartbeat_at == later


def test_finish_run_terminal_clears_ownership_closes_execution(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    claim_ownership(run, pid=1, host="h1", now=FIXED)
    run.record_execution(ExecutionRecord(execution_id="exec-r1-1", started_at=FIXED))

    done_at = FIXED + timedelta(minutes=5)
    s = finish_run(run, RunStatus.FAILED, now=done_at)
    assert s.status == RunStatus.FAILED
    assert s.finished_at == done_at
    assert s.owner_pid is None
    assert s.owner_host is None
    assert s.heartbeat_at is None

    rec = s.executions[-1]
    assert rec.execution_id == "exec-r1-1"
    assert rec.finished_at == done_at
    assert rec.status == "failed"


# ── should_reap pure predicate (ac-004) ──────────────────────────────────────


def test_should_reap_table() -> None:
    stale = FIXED + timedelta(seconds=HEARTBEAT_STALE_SECONDS + 1)

    # not running → never reaped
    assert not should_reap(
        RunOpsState(status=RunStatus.PENDING), now=FIXED, current_host="h", pid_alive=False
    )
    # same host, dead pid → reap
    same = RunOpsState(status=RunStatus.RUNNING, owner_host="h", owner_pid=1, heartbeat_at=FIXED)
    assert should_reap(same, now=FIXED, current_host="h", pid_alive=False)
    # same host, live pid → keep
    assert not should_reap(same, now=FIXED, current_host="h", pid_alive=True)
    # cross host, stale heartbeat → reap
    cross = RunOpsState(status=RunStatus.RUNNING, owner_host="other", heartbeat_at=FIXED)
    assert should_reap(cross, now=stale, current_host="h", pid_alive=None)
    # cross host, fresh heartbeat → keep
    assert not should_reap(cross, now=FIXED, current_host="h", pid_alive=None)
    # cross host, absent heartbeat → keep (live HPC job)
    assert not should_reap(
        RunOpsState(status=RunStatus.RUNNING, owner_host="other"),
        now=FIXED,
        current_host="h",
        pid_alive=None,
    )


# ── reap_run_if_stale (ac-005) ───────────────────────────────────────────────


def test_reap_run_if_stale_flips_dead_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(name="r1", root=tmp_path)
    claim_ownership(run, pid=999, host="thishost", now=FIXED)
    monkeypatch.setattr(rl, "_pid_alive", lambda _pid: False)
    assert reap_run_if_stale(run, current_host="thishost", now=FIXED) is True
    assert run.read_ops().status == RunStatus.FAILED


def test_reap_run_if_stale_keeps_live_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(name="r2", root=tmp_path)
    claim_ownership(run, pid=999, host="thishost", now=FIXED)
    monkeypatch.setattr(rl, "_pid_alive", lambda _pid: True)
    assert reap_run_if_stale(run, current_host="thishost", now=FIXED) is False
    assert run.read_ops().status == RunStatus.RUNNING


# ── RunHeartbeat daemon (ac-006) ─────────────────────────────────────────────


def test_run_heartbeat_restamps_then_stops(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    hb = RunHeartbeat(run, interval=0.01)
    hb.start()
    try:
        beat = None
        for _ in range(300):  # poll up to ~3s
            beat = run.read_ops().heartbeat_at
            if beat is not None:
                break
            time.sleep(0.01)
        assert beat is not None  # daemon stamped a heartbeat
    finally:
        hb.stop()
    assert not hb.is_alive()


def test_run_heartbeat_context_manager(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    with RunHeartbeat(run, interval=0.01) as hb:
        assert hb.is_alive()
    assert not hb.is_alive()
