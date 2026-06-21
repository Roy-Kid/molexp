"""Tests for the resume / rerun / cancel verbs (okf-05-03).

Pure knowledge-layer re-execution semantics on okf-04 ``RunOpsState``: resume
reopens the last non-succeeded execution (same exec_id), rerun opens a fresh
``exec-{run_id}-N``, cancel flips to CANCELLED. resume/rerun are gated to the
retryable domain (failed/cancelled).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.knowledge import (
    ExecutionRecord,
    Run,
    RunNotRetryableError,
    RunStatus,
    cancel_run,
    claim_ownership,
    finish_run,
    make_execution_id,
    rerun_run,
    resumable_execution_id,
    resume_run,
)
from molexp.knowledge.ops import RunOpsState

FIXED = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


def _failed_run_with_execution(tmp_path: Path, name: str = "r1") -> Run:
    run = Run(name=name, root=tmp_path)
    claim_ownership(run, pid=1, host="h1", execution_id=f"exec-{name}", now=FIXED)
    run.record_execution(ExecutionRecord(execution_id=f"exec-{name}", started_at=FIXED))
    finish_run(run, RunStatus.FAILED, now=FIXED)
    return run


# ── pure helpers (ac-002 / ac-003) ───────────────────────────────────────────


def test_make_execution_id_format() -> None:
    assert make_execution_id("r1", 1) == "exec-r1"
    assert make_execution_id("r1", 2) == "exec-r1-2"
    assert make_execution_id("r1", 3) == "exec-r1-3"


def test_resumable_execution_id() -> None:
    assert resumable_execution_id(RunOpsState()) is None
    state = RunOpsState(
        executions=(
            ExecutionRecord(execution_id="exec-r1", started_at=FIXED, status="succeeded"),
            ExecutionRecord(execution_id="exec-r1-2", started_at=FIXED, status="failed"),
        )
    )
    assert resumable_execution_id(state) == "exec-r1-2"
    all_ok = RunOpsState(
        executions=(ExecutionRecord(execution_id="exec-r1", started_at=FIXED, status="succeeded"),)
    )
    assert resumable_execution_id(all_ok) is None


# ── resume / rerun (ac-004 / ac-005) ─────────────────────────────────────────


def test_resume_reopens_last_execution(tmp_path: Path) -> None:
    run = _failed_run_with_execution(tmp_path)
    exec_id = resume_run(run, now=FIXED)
    assert exec_id == "exec-r1"  # same execution reopened
    state = run.read_ops()
    assert state.status == RunStatus.RUNNING
    assert state.finished_at is None
    rec = next(r for r in state.executions if r.execution_id == "exec-r1")
    assert rec.finished_at is None
    assert rec.status == "running"
    assert len(state.executions) == 1  # reopened, not appended


def test_rerun_opens_fresh_execution(tmp_path: Path) -> None:
    run = _failed_run_with_execution(tmp_path)
    exec_id = rerun_run(run, now=FIXED)
    assert exec_id == "exec-r1-2"  # fresh, distinct from the original
    state = run.read_ops()
    assert state.status == RunStatus.RUNNING
    assert state.current_execution_id == "exec-r1-2"
    assert len(state.executions) == 2
    assert state.executions[-1].status == "running"


# ── cancel (ac-006) ──────────────────────────────────────────────────────────


def test_cancel_run_flips_and_clears(tmp_path: Path) -> None:
    run = Run(name="r1", root=tmp_path)
    claim_ownership(run, pid=1, host="h1", execution_id="exec-r1", now=FIXED)
    run.record_execution(ExecutionRecord(execution_id="exec-r1", started_at=FIXED))

    s = cancel_run(run, now=FIXED)
    assert s.status == RunStatus.CANCELLED
    assert s.finished_at == FIXED
    assert s.owner_pid is None
    assert s.owner_host is None
    assert s.heartbeat_at is None
    assert s.executions[-1].finished_at == FIXED
    assert s.executions[-1].status == "cancelled"


# ── retryable gating (ac-007) ────────────────────────────────────────────────


@pytest.mark.parametrize("status", [RunStatus.PENDING, RunStatus.SUCCEEDED, RunStatus.RUNNING])
def test_resume_rerun_gated_to_retryable(tmp_path: Path, status: RunStatus) -> None:
    run = Run(name="r1", root=tmp_path)
    run.write_ops(RunOpsState(status=status))
    with pytest.raises(RunNotRetryableError):
        resume_run(run, now=FIXED)
    with pytest.raises(RunNotRetryableError):
        rerun_run(run, now=FIXED)
