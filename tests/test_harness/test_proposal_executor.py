"""Guarded-execution slice 01 — the proposal dispatch spine.

RED-first: ``molexp.harness.actions`` does not exist yet. Exercises the
``ProposalExecutor`` dispatch contract with a local no-op handler and zero
curation code (curation handlers are slice 02).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import pytest

from molexp.harness.schemas import EventType
from molexp.harness.schemas.change_proposal import (
    ChangeProposal,
    ChangeSpec,
    ObjectRef,
    ProposalOutcome,
    StateSnapshot,
)

_NOW = datetime(2026, 7, 1, tzinfo=UTC)


def _ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "harness.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db)
    lineage = SQLiteArtifactLineageStore(path=db, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-ge",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
    )


def _proposal(op: str = "asset_move", affected: list[ObjectRef] | None = None) -> ChangeProposal:
    affected = affected if affected is not None else [ObjectRef(kind="run", id="r1")]
    return ChangeProposal(
        id="cp-ge-1",
        intent="relocate run r1",
        current_state=StateSnapshot(objects=list(affected)),
        proposed_change=ChangeSpec(op=op, summary="move r1", payload={}),
        affected_objects=list(affected),
        expected_benefit="tidier tree",
        risks=[],
        reversibility="reversible",
        approval_level="user",
        evidence=[],
        knowledge=[],
    )


class _NoopHandler:
    """A handler that succeeds without touching anything."""

    async def apply(self, ctx, proposal: ChangeProposal) -> ProposalOutcome:
        return ProposalOutcome(
            status="executed", decided_by="noop", decided_at=_NOW, result_artifact_ids=["art-1"]
        )


class _BoomHandler:
    """A handler that raises a runtime error inside apply."""

    async def apply(self, ctx, proposal: ChangeProposal) -> ProposalOutcome:
        raise RuntimeError("boom")


def _executor_with(op: str, handler):
    from molexp.harness.actions import ChangeActionRegistry, ProposalExecutor

    registry = ChangeActionRegistry()
    registry.register(op, handler)
    return ProposalExecutor(registry)


def test_dispatch_returns_executed_outcome(tmp_path: Path) -> None:
    """ac-001 — a registered handler yields an executed outcome + tool_* events."""
    ctx = _ctx(tmp_path)
    executor = _executor_with("asset_move", _NoopHandler())
    outcome = asyncio.run(executor.dispatch(ctx, _proposal("asset_move")))
    assert outcome.status == "executed"
    types = [e.type for e in ctx.event_log.list_events("run-ge")]
    assert types == ["tool_called", "tool_completed"]


def test_unknown_op_raises_unhandled(tmp_path: Path) -> None:
    """ac-002 — an op with no registered handler fails loudly, records no event."""
    from molexp.harness.actions import ChangeActionRegistry, ProposalExecutor
    from molexp.harness.errors import UnhandledHighRiskOpError

    ctx = _ctx(tmp_path)
    executor = ProposalExecutor(ChangeActionRegistry())  # empty registry
    with pytest.raises(UnhandledHighRiskOpError):
        asyncio.run(executor.dispatch(ctx, _proposal("workflow_change")))
    assert ctx.event_log.list_events("run-ge") == []


def test_handler_raise_recorded_as_failed(tmp_path: Path) -> None:
    """ac-003 — a handler exception becomes status=failed + tool_failed, not raised."""
    ctx = _ctx(tmp_path)
    executor = _executor_with("asset_move", _BoomHandler())
    outcome = asyncio.run(executor.dispatch(ctx, _proposal("asset_move")))
    assert outcome.status == "failed"
    assert outcome.reason and "boom" in outcome.reason
    types = [e.type for e in ctx.event_log.list_events("run-ge")]
    assert types == ["tool_called", "tool_failed"]


def test_out_of_affected_scope_rejected(tmp_path: Path) -> None:
    """ac-004 — assert_within_affected_scope guards the §8.2 binding scope."""
    from molexp.harness.actions import assert_within_affected_scope
    from molexp.harness.errors import OutOfAffectedScopeError

    proposal = _proposal("asset_move", affected=[ObjectRef(kind="run", id="r1")])
    # in-scope target → returns None
    assert assert_within_affected_scope(proposal, [ObjectRef(kind="run", id="r1")]) is None
    # out-of-scope target → raises
    with pytest.raises(OutOfAffectedScopeError):
        assert_within_affected_scope(proposal, [ObjectRef(kind="run", id="ghost")])

    # a handler that hits an out-of-scope target is recorded status=failed + tool_failed
    class _ScopeViolating:
        async def apply(self, ctx, prop):
            assert_within_affected_scope(prop, [ObjectRef(kind="run", id="ghost")])
            return ProposalOutcome(status="executed", decided_by="x", decided_at=_NOW)

    ctx = _ctx(tmp_path)
    executor = _executor_with("asset_move", _ScopeViolating())
    outcome = asyncio.run(executor.dispatch(ctx, proposal))
    assert outcome.status == "failed"
    assert [e.type for e in ctx.event_log.list_events("run-ge")] == ["tool_called", "tool_failed"]


def test_action_events_carry_proposal_and_op(tmp_path: Path) -> None:
    """ac-005 — every action event carries proposal_id + high_risk_op (invariant #7)."""
    ctx = _ctx(tmp_path)
    executor = _executor_with("asset_move", _NoopHandler())
    asyncio.run(executor.dispatch(ctx, _proposal("asset_move")))
    for event in ctx.event_log.list_events("run-ge"):
        assert event.payload["proposal_id"] == "cp-ge-1"
        assert event.payload["high_risk_op"] == "asset_move"


def test_public_surface_and_eventtype_not_widened() -> None:
    """ac-006 — the actions surface exports and EventType is not widened."""
    from molexp.harness.actions import ChangeActionHandler, ProposalActionRecorder, ProposalExecutor
    from molexp.harness.errors import OutOfAffectedScopeError, UnhandledHighRiskOpError

    assert ProposalExecutor and ChangeActionHandler and ProposalActionRecorder
    assert UnhandledHighRiskOpError and OutOfAffectedScopeError
    # EventType still carries exactly its pre-existing tool_* members — no new ones.
    members = set(get_args(EventType))
    assert {"tool_called", "tool_completed", "tool_failed"} <= members
    assert "action_completed" not in members
    assert "action_failed" not in members
