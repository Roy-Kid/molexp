"""curate-unify-01 — the shared ChangeProposal curation backend.

RED-first: ``molexp.services.curate_runtime.proposal_flow`` does not exist yet.
Exercises the pure `capability → ChangeProposal` mapping and the
`run_curation_proposal` backend (gate + executor, single approval) over a real
on-disk Workspace.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.change_proposal_gate import approval_level_for
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.schemas.change_proposal import ObjectRef
from molexp.workspace import Workspace


def _workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path, name="Lab")
    project = ws.add_project("p")
    project.add_experiment("e1", workflow_source="t.py")
    project.add_experiment("e2", workflow_source="t.py")
    project.get_experiment("e1").add_run(id="r1")
    return ws


def _audit_run(ws: Workspace):
    return ws.add_project("curations").add_experiment("curate").add_run(id="audit1")


async def _reject(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="tester",
        decided_at=datetime(2026, 7, 1, tzinfo=UTC),
        reason="no",
    )


# ── mapping ──────────────────────────────────────────────────────────────────


def test_read_only_cap_maps_to_none() -> None:
    """ac-001 — read-only curation caps map to None (no proposal)."""
    from molexp.services.curate_runtime import curation_invocation_to_proposal

    for cap_id in (
        "molexp.curation.scan_workspace",
        "molexp.curation.find_asset_by_hash",
        "molexp.curation.aggregate_assets_by_kind",
        "molexp.curation.dedupe_workflow_source",
        "molexp.curation.consolidate_workflow_source",
    ):
        assert curation_invocation_to_proposal(cap_id, {}) is None


def test_move_run_mapping() -> None:
    """ac-002 — move_run invocation maps to a valid asset_move ChangeProposal."""
    from molexp.services.curate_runtime import curation_invocation_to_proposal

    proposal = curation_invocation_to_proposal(
        "molexp.curation.move_run", {"run": "r1", "target_experiment": "e2"}
    )
    assert proposal is not None
    assert proposal.proposed_change.op == "asset_move"
    assert proposal.affected_objects == [
        ObjectRef(kind="run", id="r1"),
        ObjectRef(kind="experiment", id="e2"),
    ]
    assert proposal.proposed_change.payload == {
        "curation_op": "move_run",
        "target_experiment": {"kind": "experiment", "id": "e2"},
    }
    assert proposal.approval_level == approval_level_for("asset_move", proposal.reversibility)


def test_delete_folder_mapping() -> None:
    """ac-003 — delete_folder invocation maps to artifact_delete targeting a run ref."""
    from molexp.services.curate_runtime import curation_invocation_to_proposal

    proposal = curation_invocation_to_proposal("molexp.curation.delete_folder", {"folder": "r1"})
    assert proposal is not None
    assert proposal.proposed_change.op == "artifact_delete"
    assert proposal.affected_objects == [ObjectRef(kind="run", id="r1")]
    assert proposal.proposed_change.payload == {}


def test_unknown_cap_raises() -> None:
    """ac-004 — an unknown / non-curation capability id fails loudly."""
    from molexp.services.curate_runtime import curation_invocation_to_proposal

    with pytest.raises(ValueError):
        curation_invocation_to_proposal("molexp.curation.not_a_thing", {})


# ── backend ──────────────────────────────────────────────────────────────────


def test_backend_move_run_grant(tmp_path: Path) -> None:
    """ac-005 — a granted move_run executes: run relocated + change_proposal artifact."""
    from molexp.services.curate_runtime import (
        curation_invocation_to_proposal,
        run_curation_proposal,
    )

    ws = _workspace(tmp_path)
    run = _audit_run(ws)
    proposal = curation_invocation_to_proposal(
        "molexp.curation.move_run", {"run": "r1", "target_experiment": "e2"}
    )
    result = asyncio.run(run_curation_proposal(proposal, workspace=ws, run=run))
    assert result.execution_result.status == "executed"
    fresh = Workspace(root=tmp_path, name="Lab")
    assert not fresh.get_project("p").get_experiment("e1").has_run("r1")
    assert fresh.get_project("p").get_experiment("e2").has_run("r1")
    assert (Path(run.run_dir) / "artifacts").exists()


def test_backend_delete_grant(tmp_path: Path) -> None:
    """ac-006 — a granted delete_folder(run) removes the run folder, executed."""
    from molexp.services.curate_runtime import (
        curation_invocation_to_proposal,
        run_curation_proposal,
    )

    ws = _workspace(tmp_path)
    run = _audit_run(ws)
    proposal = curation_invocation_to_proposal("molexp.curation.delete_folder", {"folder": "r1"})
    result = asyncio.run(run_curation_proposal(proposal, workspace=ws, run=run))
    assert result.execution_result.status == "executed"
    assert (
        not Workspace(root=tmp_path, name="Lab").get_project("p").get_experiment("e1").has_run("r1")
    )


def test_backend_reject(tmp_path: Path) -> None:
    """ac-007 — a rejected proposal performs no mutation, records rejected."""
    from molexp.services.curate_runtime import (
        curation_invocation_to_proposal,
        run_curation_proposal,
    )

    ws = _workspace(tmp_path)
    run = _audit_run(ws)
    proposal = curation_invocation_to_proposal(
        "molexp.curation.move_run", {"run": "r1", "target_experiment": "e2"}
    )
    result = asyncio.run(run_curation_proposal(proposal, workspace=ws, run=run, approve=_reject))
    assert result.execution_result.status == "rejected"
    assert Workspace(root=tmp_path, name="Lab").get_project("p").get_experiment("e1").has_run("r1")
