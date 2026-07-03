"""Guarded-execution slice 03 — wire ProposalExecutor into the ChangeProposal gate.

Guarded-path tests only; the dry-gate tests in ``test_change_proposal.py`` are the
untouched regression net. Exercises the opt-in ``executor=`` param on
``gate_change_proposal``: granted → dispatch (executed/failed), reject → dry,
single-gate, no double side-effect approval.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.change_proposal_gate import gate_change_proposal
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.schemas.change_proposal import (
    ChangeProposal,
    ChangeSpec,
    ObjectRef,
    StateSnapshot,
)
from molexp.harness.stages.approval_gate import auto_grant_approver
from molexp.workspace import Workspace
from molexp.workspace.folder import Folder

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
        run_id="run-ge3",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
    )


def _workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path, name="Lab")
    project = ws.add_project("p")
    project.add_experiment("e1", workflow_source="t.py")
    project.add_experiment("e2", workflow_source="t.py")
    project.get_experiment("e1").add_run(id="r1")
    ws.add_folder(Folder(parent=ws, name="scratch", kind="curation.test"))
    return ws


def _move_run_proposal() -> ChangeProposal:
    affected = [ObjectRef(kind="run", id="r1"), ObjectRef(kind="experiment", id="e2")]
    return _proposal(
        "asset_move",
        affected,
        {"curation_op": "move_run", "target_experiment": {"kind": "experiment", "id": "e2"}},
    )


def _proposal(op: str, affected: list[ObjectRef], payload: dict) -> ChangeProposal:
    return ChangeProposal(
        id="cp-ge3",
        intent=f"{op} action",
        current_state=StateSnapshot(objects=list(affected)),
        proposed_change=ChangeSpec(op=op, summary=op, payload=payload),
        affected_objects=list(affected),
        expected_benefit="tidier tree",
        risks=[],
        reversibility="reversible",
        approval_level="user",
        evidence=[],
        knowledge=[],
    )


def _curation_executor():
    from molexp.harness.actions import ChangeActionRegistry, ProposalExecutor
    from molexp.harness.actions.handlers import register_curation_handlers

    registry = ChangeActionRegistry()
    register_curation_handlers(registry)
    return ProposalExecutor(registry)


async def _reject(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="tester",
        decided_at=_NOW,
        reason="no",
    )


def _run_in(ws_root: Path, experiment: str) -> bool:
    ws = Workspace(root=ws_root, name="Lab")
    return ws.get_project("p").get_experiment(experiment).has_run("r1")


# ── ac-001: dry default preserved ────────────────────────────────────────────


def test_dry_default_no_executor(tmp_path: Path) -> None:
    """ac-001 — executor=None keeps the advisory dry behavior (granted, no run)."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    result = asyncio.run(
        gate_change_proposal(ctx, _move_run_proposal(), approve=auto_grant_approver)
    )
    assert result.execution_result is not None
    assert result.execution_result.status == "granted"
    assert _run_in(tmp_path, "e1")  # not executed → run stayed put
    assert not _run_in(tmp_path, "e2")


# ── ac-002 / ac-003 / ac-004: granted + executor ─────────────────────────────


def test_granted_executed_move_run(tmp_path: Path) -> None:
    """ac-002 — granted asset_move + executor performs the move + reports executed."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    result = asyncio.run(
        gate_change_proposal(
            ctx, _move_run_proposal(), approve=auto_grant_approver, executor=_curation_executor()
        )
    )
    assert result.execution_result.status == "executed"
    assert result.execution_result.result_artifact_ids
    assert _run_in(tmp_path, "e2")
    assert not _run_in(tmp_path, "e1")


def test_granted_executed_artifact_delete(tmp_path: Path) -> None:
    """ac-002 — granted artifact_delete + executor deletes the folder."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    proposal = _proposal("artifact_delete", [ObjectRef(kind="folder", id="scratch")], {})
    result = asyncio.run(
        gate_change_proposal(
            ctx, proposal, approve=auto_grant_approver, executor=_curation_executor()
        )
    )
    assert result.execution_result.status == "executed"
    assert not Workspace(root=tmp_path, name="Lab").has_folder("scratch", cls=Folder)


def test_timeline_requested_granted_completed(tmp_path: Path) -> None:
    """ac-003 — the event timeline is approval_requested → approval_granted → tool_completed."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    asyncio.run(
        gate_change_proposal(
            ctx, _move_run_proposal(), approve=auto_grant_approver, executor=_curation_executor()
        )
    )
    types = [e.type for e in ctx.event_log.list_events("run-ge3")]
    assert "approval_requested" in types
    assert "approval_granted" in types
    assert "tool_completed" in types
    assert (
        types.index("approval_requested")
        < types.index("approval_granted")
        < types.index("tool_completed")
    )


def test_single_gate_one_approval_round(tmp_path: Path) -> None:
    """ac-004 — the mutation is gated exactly once (one approval round, one move)."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    asyncio.run(
        gate_change_proposal(
            ctx, _move_run_proposal(), approve=auto_grant_approver, executor=_curation_executor()
        )
    )
    types = [e.type for e in ctx.event_log.list_events("run-ge3")]
    assert types.count("approval_requested") == 1
    assert types.count("approval_granted") == 1
    # applied once: run present in e2, gone from e1
    assert _run_in(tmp_path, "e2")
    assert not _run_in(tmp_path, "e1")


# ── ac-005: reject stays dry ─────────────────────────────────────────────────


def test_reject_with_executor_stays_dry(tmp_path: Path) -> None:
    """ac-005 — a rejected proposal performs no mutation; the proposal is preserved."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    result = asyncio.run(
        gate_change_proposal(
            ctx, _move_run_proposal(), approve=_reject, executor=_curation_executor()
        )
    )
    assert result.execution_result.status == "rejected"
    assert _run_in(tmp_path, "e1")  # no move
    # no action ever dispatched
    assert not any(e.type.startswith("tool_") for e in ctx.event_log.list_events("run-ge3"))
    # the change_proposal artifact is preserved (§8.3)
    preserved_id = result.execution_result.result_artifact_ids[0]
    assert ctx.artifact_store.get_ref(preserved_id) is not None


# ── ac-006: runtime failure recorded ─────────────────────────────────────────


def test_runtime_failure_recorded_not_raised(tmp_path: Path) -> None:
    """ac-006 — a reorg runtime failure yields status=failed, gate does not raise."""
    ws = _workspace(tmp_path)
    ws.get_project("p").get_experiment("e2").add_run(id="r1")  # collision on move
    ctx = _ctx(tmp_path)
    result = asyncio.run(
        gate_change_proposal(
            ctx, _move_run_proposal(), approve=auto_grant_approver, executor=_curation_executor()
        )
    )
    assert result.execution_result.status == "failed"


# ── ac-007: fail-loud ────────────────────────────────────────────────────────


def test_unknown_op_propagates_past_gate(tmp_path: Path) -> None:
    """ac-007 — an unknown-op proposal raises UnhandledHighRiskOpError past the gate."""
    from molexp.harness.errors import UnhandledHighRiskOpError

    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    # workflow_change has no registered curation handler
    proposal = _proposal("workflow_change", [ObjectRef(kind="workflow", id="wf-1")], {})
    with pytest.raises(UnhandledHighRiskOpError):
        asyncio.run(
            gate_change_proposal(
                ctx, proposal, approve=auto_grant_approver, executor=_curation_executor()
            )
        )


def test_out_of_bounds_recorded_failed_no_mutation(tmp_path: Path) -> None:
    """ac-007 — an out-of-affected_objects target is recorded failed, run unmutated."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    # affected omits e2 but the payload names it
    proposal = _proposal(
        "asset_move",
        [ObjectRef(kind="run", id="r1")],
        {"curation_op": "move_run", "target_experiment": {"kind": "experiment", "id": "e2"}},
    )
    result = asyncio.run(
        gate_change_proposal(
            ctx, proposal, approve=auto_grant_approver, executor=_curation_executor()
        )
    )
    assert result.execution_result.status == "failed"
    assert _run_in(tmp_path, "e1")  # unmutated
