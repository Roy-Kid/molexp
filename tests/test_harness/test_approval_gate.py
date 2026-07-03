"""``ApprovalGate`` store-first semantics (spec vision-loop-01-approval-inbox).

There is **no auto-grant default** anymore. ``approve=None`` means, per request:

1. a stored grant in ``ctx.approval_store`` passes immediately (the stored
   decision is re-logged, ``decided_by`` preserved);
2. otherwise an injected approver is asked live and its decision is persisted
   to the store *and* the event log;
3. otherwise the request is recorded pending and the gate raises
   ``ApprovalPendingError`` — suspension, not failure, and **no** summary
   artifact is persisted.

Audit contract stays intact in all three paths: ``approval_requested`` is
recorded before any decision event (the ``audit``-named tests pin the order;
select them with ``pytest -k audit``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import ApprovalPendingError, StageExecutionError
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.stages import ApprovalGate
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

_RUN_ID = "run-gate"


def _request(request_id: str = "req-gate-1") -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id,
        intent="experiment_spec",
        reason="approve the concrete experiment spec",
        triggered_by_policy="PlanMode",
        created_at=datetime(2026, 7, 3, tzinfo=UTC),
    )


def _decision(
    request_id: str,
    *,
    granted: bool,
    decided_by: str,
    reason: str | None = None,
) -> ApprovalDecision:
    return ApprovalDecision(
        request_id=request_id,
        granted=granted,
        decided_by=decided_by,
        decided_at=datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
        reason=reason,
    )


class _Env:
    """One run's stores + ctx, all sharing ``harness.sqlite`` like Mode._build_ctx."""

    def __init__(self, tmp_path: Path) -> None:
        db = tmp_path / "harness.sqlite"
        self.artifacts = FileArtifactStore(root=tmp_path / "artifacts")
        self.event_log = SQLiteEventLog(path=db)
        self.approval_store = SQLiteApprovalStore(db)
        self.ctx = HarnessRunContext(
            run_id=_RUN_ID,
            workspace_root=tmp_path,
            artifact_store=self.artifacts,
            event_log=self.event_log,
            lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=self.artifacts),
            approval_store=self.approval_store,
        )

    def event_types(self) -> list[str]:
        return [event.type for event in self.event_log.list_events(_RUN_ID)]


@pytest.fixture()
def env(tmp_path: Path) -> _Env:
    return _Env(tmp_path)


async def _grant_live(request: ApprovalRequest) -> ApprovalDecision:
    return _decision(request.id, granted=True, decided_by="live-human", reason="looks right")


async def _reject_live(request: ApprovalRequest) -> ApprovalDecision:
    return _decision(request.id, granted=False, decided_by="live-human", reason="nope")


# ─────────────────────────────────── pending path (approve=None, no grant)


class TestPendingSuspension:
    async def test_no_approver_and_no_stored_grant_raises_approval_pending(self, env: _Env) -> None:
        request = _request()
        gate = ApprovalGate(requests=[request], approve=None)

        with pytest.raises(ApprovalPendingError) as excinfo:
            await gate.run(env.ctx)

        assert excinfo.value.run_id == _RUN_ID
        assert [r.id for r in excinfo.value.requests] == [request.id]

    async def test_pending_suspension_records_pending_and_no_artifact(self, env: _Env) -> None:
        request = _request()
        gate = ApprovalGate(requests=[request], approve=None)

        with pytest.raises(ApprovalPendingError):
            await gate.run(env.ctx)

        [pending] = env.approval_store.pending(_RUN_ID)
        assert pending.id == request.id
        # Suspension persists NO summary artifact — nothing was approved.
        assert env.artifacts.latest_by_kind("analysis_result") is None

    async def test_pending_error_is_a_stage_execution_error(self) -> None:
        # Existing `except StageExecutionError` call sites must still stop.
        assert issubclass(ApprovalPendingError, StageExecutionError)

    async def test_audit_pending_records_the_request_and_no_decision(self, env: _Env) -> None:
        gate = ApprovalGate(requests=[_request()], approve=None)

        with pytest.raises(ApprovalPendingError):
            await gate.run(env.ctx)

        types = env.event_types()
        assert types == ["approval_requested"]  # asked, nobody decided (yet)


# ───────────────────────────────────────── stored-grant replay path


class TestStoredGrantReplay:
    def _store_grant(self, env: _Env, request: ApprovalRequest) -> None:
        env.approval_store.record_pending(_RUN_ID, request)
        env.approval_store.record_decision(
            _decision(request.id, granted=True, decided_by="ui-operator", reason="approved")
        )

    async def test_stored_grant_passes_the_gate(self, env: _Env) -> None:
        request = _request()
        self._store_grant(env, request)
        gate = ApprovalGate(requests=[request], approve=None)

        ref = await gate.run(env.ctx)

        assert ref.kind == "analysis_result"

    async def test_audit_stored_replay_logs_request_then_decision_preserving_decided_by(
        self, env: _Env
    ) -> None:
        request = _request()
        self._store_grant(env, request)
        gate = ApprovalGate(requests=[request], approve=None)

        await gate.run(env.ctx)

        events = env.event_log.list_events(_RUN_ID)
        assert [e.type for e in events] == ["approval_requested", "approval_granted"]
        # The audit shows the re-entry resolved from the stored grant.
        assert events[-1].payload["decided_by"] == "ui-operator"


# ─────────────────────────────────────────── live-approver path


class TestInjectedApprover:
    async def test_injected_approver_grants_and_persists_to_store(self, env: _Env) -> None:
        request = _request()
        gate = ApprovalGate(requests=[request], approve=_grant_live)

        ref = await gate.run(env.ctx)

        assert ref.kind == "analysis_result"
        stored = env.approval_store.granted_decision_for(request.id)
        assert stored is not None
        assert stored.decided_by == "live-human"
        assert env.approval_store.pending(_RUN_ID) == []

    async def test_injected_approver_rejection_fails_and_persists(self, env: _Env) -> None:
        request = _request()
        gate = ApprovalGate(requests=[request], approve=_reject_live)

        with pytest.raises(StageExecutionError) as excinfo:
            await gate.run(env.ctx)

        assert not isinstance(excinfo.value, ApprovalPendingError)  # rejection ≠ suspension
        # Recorded as decided-rejected: not pending, and never a replayable grant.
        assert env.approval_store.granted_decision_for(request.id) is None
        assert env.approval_store.pending(_RUN_ID) == []

    async def test_audit_live_approver_logs_request_then_decision(self, env: _Env) -> None:
        gate = ApprovalGate(requests=[_request()], approve=_grant_live)

        await gate.run(env.ctx)

        types = env.event_types()
        assert types == ["approval_requested", "approval_granted"]

    async def test_audit_live_rejection_logs_request_then_rejection(self, env: _Env) -> None:
        gate = ApprovalGate(requests=[_request()], approve=_reject_live)

        with pytest.raises(StageExecutionError):
            await gate.run(env.ctx)

        types = env.event_types()
        assert types == ["approval_requested", "approval_rejected"]
