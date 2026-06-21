"""Tests for the typed ``_ops/`` operational sidecar (okf-04).

A Run's hot machine state (status / owner / timestamps / heartbeat /
executions) lives in ``_ops/run.json`` — typed via :class:`RunOpsState`,
physically isolated from the knowledge-layer ``meta.yaml``. Timestamps are
aware-UTC so heartbeat staleness compares correctly across hosts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import molexp.atomicio as atomicio
from molexp.knowledge import (
    RETRYABLE_STATUSES,
    ConceptMeta,
    ExecutionRecord,
    Run,
    RunOpsState,
    RunStatus,
)
from molexp.knowledge.ops import HEARTBEAT_STALE_SECONDS, RUN_OPS_NAME


def _ops_path(run: Run) -> Path:
    return Path(run.resolve()) / "_ops" / f"{RUN_OPS_NAME}.json"


# ── models (ac-002 / ac-003) ─────────────────────────────────────────────────


def test_run_status_values() -> None:
    assert {s.value for s in RunStatus} == {
        "pending",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    }


def test_retryable_statuses() -> None:
    assert frozenset({"failed", "cancelled"}) == RETRYABLE_STATUSES


def test_is_retryable() -> None:
    assert RunOpsState(status=RunStatus.FAILED).is_retryable
    assert RunOpsState(status=RunStatus.CANCELLED).is_retryable
    assert not RunOpsState(status=RunStatus.RUNNING).is_retryable
    assert not RunOpsState().is_retryable  # default pending


def test_default_state() -> None:
    s = RunOpsState()
    assert s.status == RunStatus.PENDING
    assert s.owner_pid is None
    assert s.owner_host is None
    assert s.started_at is None
    assert s.finished_at is None
    assert s.heartbeat_at is None
    assert s.current_execution_id is None
    assert s.executions == ()


def test_frozen() -> None:
    s = RunOpsState()
    with pytest.raises(ValidationError):
        s.status = RunStatus.RUNNING  # type: ignore[misc]


def test_json_round_trip_preserves_aware_utc_and_executions() -> None:
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    rec = ExecutionRecord(execution_id="exec-r1", started_at=now)
    s = RunOpsState(
        status=RunStatus.RUNNING,
        owner_pid=123,
        owner_host="node1",
        started_at=now,
        heartbeat_at=now,
        current_execution_id="exec-r1",
        executions=(rec,),
    )
    back = RunOpsState.model_validate(s.model_dump(mode="json"))
    assert back == s
    assert back.executions[0].execution_id == "exec-r1"
    assert back.started_at == now  # aware UTC preserved


# ── Run typed accessors (ac-004 / ac-005 / ac-006) ───────────────────────────


def test_read_ops_default_when_absent(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    s = run.read_ops()
    assert isinstance(s, RunOpsState)
    assert s.status == RunStatus.PENDING


def test_write_ops_persists_to_ops_run_json(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    run.write_ops(RunOpsState(status=RunStatus.RUNNING, owner_pid=7))
    assert _ops_path(run).is_file()
    assert run.read_ops().status == RunStatus.RUNNING
    assert run.read_ops().owner_pid == 7


def test_update_ops_typed_rmw_under_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = Run(name="r1", root=tmp_path)
    run.write_ops(RunOpsState(status=RunStatus.PENDING))

    used = {"lock": False}
    real_lock = atomicio.file_lock

    def spy_lock(path: Path, **kw: object):
        used["lock"] = True
        return real_lock(path, **kw)

    monkeypatch.setattr(atomicio, "file_lock", spy_lock)

    out = run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.RUNNING}))
    assert isinstance(out, RunOpsState)
    assert out.status == RunStatus.RUNNING
    assert run.read_ops().status == RunStatus.RUNNING
    assert used["lock"] is True


def test_ops_writes_isolated_from_meta(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    run.write_meta(ConceptMeta(type="run", id="r1"))
    before = (Path(run.resolve()) / "meta.yaml").read_text()

    run.set_status(RunStatus.RUNNING, now=datetime(2026, 6, 21, tzinfo=UTC))
    run.beat(now=datetime(2026, 6, 21, tzinfo=UTC))

    after = (Path(run.resolve()) / "meta.yaml").read_text()
    assert before == after  # hot state never touched meta.yaml
    assert _ops_path(run).is_file()


# ── mutators (ac-007) ────────────────────────────────────────────────────────


def test_set_status_terminal_sets_finished_at(tmp_path: Path) -> None:
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    run = Run(name="r1", root=tmp_path)
    s = run.set_status(RunStatus.SUCCEEDED, now=now)
    assert s.status == RunStatus.SUCCEEDED
    assert s.finished_at == now


def test_set_status_running_sets_started_clears_finished(tmp_path: Path) -> None:
    t0 = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)
    run = Run(name="r1", root=tmp_path)
    run.set_status(RunStatus.FAILED, now=t0)  # sets finished_at
    s = run.set_status(RunStatus.RUNNING, now=t1)
    assert s.started_at == t1
    assert s.finished_at is None


def test_beat_refreshes_heartbeat(tmp_path: Path) -> None:
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    run = Run(name="r1", root=tmp_path)
    s = run.beat(now=now)
    assert s.heartbeat_at == now


def test_record_execution_appends_and_sets_current(tmp_path: Path) -> None:
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    run = Run(name="r1", root=tmp_path)
    s = run.record_execution(ExecutionRecord(execution_id="exec-r1-1", started_at=now))
    assert s.current_execution_id == "exec-r1-1"
    assert s.executions[-1].execution_id == "exec-r1-1"

    s2 = run.record_execution(ExecutionRecord(execution_id="exec-r1-2", started_at=now))
    assert len(s2.executions) == 2
    assert s2.current_execution_id == "exec-r1-2"


def test_is_retryable_accessor(tmp_path: Path) -> None:
    now = datetime(2026, 6, 21, tzinfo=UTC)
    run = Run(name="r1", root=tmp_path)
    run.set_status(RunStatus.FAILED, now=now)
    assert run.is_retryable()
    run.set_status(RunStatus.RUNNING, now=now)
    assert not run.is_retryable()


# ── heartbeat staleness (ac-008) ─────────────────────────────────────────────


def test_heartbeat_age_none_when_never_beaten() -> None:
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    assert RunOpsState().heartbeat_age(now) is None


def test_fresh_heartbeat_not_stale() -> None:
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    s = RunOpsState(heartbeat_at=now)
    assert not s.is_heartbeat_stale(now)
    assert s.heartbeat_age(now) == timedelta(0)


def test_old_heartbeat_is_stale() -> None:
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    old = now - timedelta(seconds=HEARTBEAT_STALE_SECONDS + 1)
    s = RunOpsState(heartbeat_at=old)
    assert s.is_heartbeat_stale(now)
