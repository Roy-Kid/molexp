"""Tests for ``molexp.harness.actions.proposal_executor`` — the dispatch spine.

Owns the ``ProposalExecutor`` dispatch contract (executed / unhandled /
recorded-failure), the ``assert_within_affected_scope`` binding guard, and the
audit invariant that action events reuse the existing ``tool_*`` vocabulary
rather than widening ``EventType``. The gated end-to-end path (grant / reject /
runtime-failure through ``gate_change_proposal``) is owned by
``test_guarded_execution.py``.
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


class TestProposalExecutor:
    def test_registered_handler_yields_executed_outcome_and_tool_events(
        self, tmp_path: Path
    ) -> None:
        ctx = _ctx(tmp_path)
        executor = _executor_with("asset_move", _NoopHandler())
        outcome = asyncio.run(executor.dispatch(ctx, _proposal("asset_move")))
        assert outcome.status == "executed"
        types = [e.type for e in ctx.event_log.list_events("run-ge")]
        assert types == ["tool_called", "tool_completed"]

    def test_unknown_op_raises_and_records_no_event(self, tmp_path: Path) -> None:
        from molexp.harness.actions import ChangeActionRegistry, ProposalExecutor
        from molexp.harness.errors import UnhandledHighRiskOpError

        ctx = _ctx(tmp_path)
        executor = ProposalExecutor(ChangeActionRegistry())  # empty registry
        with pytest.raises(UnhandledHighRiskOpError):
            asyncio.run(executor.dispatch(ctx, _proposal("workflow_change")))
        assert ctx.event_log.list_events("run-ge") == []

    def test_handler_exception_recorded_as_failed_not_raised(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        executor = _executor_with("asset_move", _BoomHandler())
        outcome = asyncio.run(executor.dispatch(ctx, _proposal("asset_move")))
        assert outcome.status == "failed"
        assert outcome.reason and "boom" in outcome.reason
        types = [e.type for e in ctx.event_log.list_events("run-ge")]
        assert types == ["tool_called", "tool_failed"]

    def test_action_events_carry_proposal_id_and_op(self, tmp_path: Path) -> None:
        """Every action event stamps proposal_id + high_risk_op (audit invariant #7)."""
        ctx = _ctx(tmp_path)
        executor = _executor_with("asset_move", _NoopHandler())
        asyncio.run(executor.dispatch(ctx, _proposal("asset_move")))
        for event in ctx.event_log.list_events("run-ge"):
            assert event.payload["proposal_id"] == "cp-ge-1"
            assert event.payload["high_risk_op"] == "asset_move"


class TestAssertWithinAffectedScope:
    def test_in_scope_target_returns_none(self) -> None:
        from molexp.harness.actions import assert_within_affected_scope

        proposal = _proposal("asset_move", affected=[ObjectRef(kind="run", id="r1")])
        assert assert_within_affected_scope(proposal, [ObjectRef(kind="run", id="r1")]) is None

    def test_out_of_scope_target_raises(self) -> None:
        from molexp.harness.actions import assert_within_affected_scope
        from molexp.harness.errors import OutOfAffectedScopeError

        proposal = _proposal("asset_move", affected=[ObjectRef(kind="run", id="r1")])
        with pytest.raises(OutOfAffectedScopeError):
            assert_within_affected_scope(proposal, [ObjectRef(kind="run", id="ghost")])


class TestActionEventVocabulary:
    def test_eventtype_reuses_tool_events_and_is_not_widened(self) -> None:
        """Guarded execution records via the existing ``tool_*`` events; EventType
        gains no ``action_*`` member."""
        members = set(get_args(EventType))
        assert {"tool_called", "tool_completed", "tool_failed"} <= members
        assert "action_completed" not in members
        assert "action_failed" not in members
