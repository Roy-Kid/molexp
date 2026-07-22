"""The shared ChangeProposal curation backend (curate-unify-01).

Mirrors :mod:`molexp.services.curate_runtime.proposal_flow`: the pure
``capability → ChangeProposal`` mapping (:func:`curation_invocation_to_proposal`)
and the gate+execute backend (:func:`run_curation_proposal`) over a real on-disk
Workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.harness.change_proposal_gate import approval_level_for
from molexp.harness.schemas.change_proposal import ObjectRef
from molexp.harness.stages.approval_gate import auto_grant_approver
from molexp.services.curate_runtime import (
    curation_invocation_to_proposal,
    run_curation_proposal,
)
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


class TestCurationInvocationToProposal:
    def test_read_only_capabilities_map_to_none(self) -> None:
        """Read-only curation caps map to None (the in-process path handles them)."""
        for cap_id in (
            "molexp.curation.scan_workspace",
            "molexp.curation.find_asset_by_hash",
            "molexp.curation.aggregate_assets_by_kind",
            "molexp.curation.dedupe_workflow_source",
            "molexp.curation.consolidate_workflow_source",
        ):
            assert curation_invocation_to_proposal(cap_id, {}) is None

    def test_move_run_maps_to_asset_move_proposal(self) -> None:
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

    def test_delete_folder_maps_to_artifact_delete_proposal(self) -> None:
        proposal = curation_invocation_to_proposal(
            "molexp.curation.delete_folder", {"folder": "r1"}
        )
        assert proposal is not None
        assert proposal.proposed_change.op == "artifact_delete"
        assert proposal.affected_objects == [ObjectRef(kind="run", id="r1")]
        assert proposal.proposed_change.payload == {}
        # An irreversible delete escalates to an elevated approval.
        assert proposal.approval_level == "elevated"

    def test_unknown_capability_raises(self) -> None:
        with pytest.raises(ValueError):
            curation_invocation_to_proposal("molexp.curation.not_a_thing", {})


class TestRunCurationProposal:
    def test_granted_delete_removes_the_run_folder(self, tmp_path: Path) -> None:
        """A granted delete_folder(run) removes the run folder, executed."""
        ws = _workspace(tmp_path)
        run = _audit_run(ws)
        proposal = curation_invocation_to_proposal(
            "molexp.curation.delete_folder", {"folder": "r1"}
        )
        result = asyncio.run(
            run_curation_proposal(proposal, workspace=ws, run=run, approve=auto_grant_approver)
        )
        assert result.execution_result.status == "executed"
        assert (
            not Workspace(root=tmp_path, name="Lab")
            .get_project("p")
            .get_experiment("e1")
            .has_run("r1")
        )
