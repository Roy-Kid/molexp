"""``curate_runtime.proposal_flow`` — the shared curation ChangeProposal mapping.

Two contracts of the services proposal mapping:

* **The five ``molexp.lifecycle.*`` ids map onto ChangeProposals** (op
  ``run_lifecycle``, verb discriminated by ``payload["lifecycle_verb"]``), each
  carrying the spec's reversibility (``irreversible`` prune, ``partially``
  execute family, ``reversible`` cancel). No known destructive capability maps
  to ``None`` — the ``CurationArgumentError`` escape is unreachable for them.
* **``rehome_asset`` is expressible in the flat reference map** via colon-encoded
  scope refs (``source: "experiment:exp-a"``, ``target: "project:demo"``).

Gating end-to-end: a proposal denied by the approver mutates NOTHING — run
status, ``_ops`` history and disk are untouched; the outcome records ``rejected``
and never raises through the flow.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, ChangeProposal
from molexp.services.curate_runtime import (
    curation_invocation_to_proposal,
    run_curation_proposal,
)
from molexp.workspace import Workspace
from molexp.workspace.models import ExecutionRecord, RunStatus
from molexp.workspace.run import Run

#: id → (lifecycle_verb, reversibility) — the spec table.
LIFECYCLE_TABLE: dict[str, tuple[str, str]] = {
    "molexp.lifecycle.run_execute": ("execute", "partially"),
    "molexp.lifecycle.run_resume": ("resume", "partially"),
    "molexp.lifecycle.run_rerun": ("rerun", "partially"),
    "molexp.lifecycle.run_cancel": ("cancel", "reversible"),
    "molexp.lifecycle.runs_prune": ("prune", "irreversible"),
}


def _workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path, name="Lab")
    project = ws.add_project("p")
    project.add_experiment("e1", workflow_source="t.py")
    project.get_experiment("e1").add_run(id="r1")
    return ws


def _target_run(ws: Workspace) -> Run:
    return ws.get_project("p").get_experiment("e1").get_run("r1")


def _audit_run(ws: Workspace) -> Run:
    return ws.add_project("curations").add_experiment("curate").add_run(id="audit1")


def _seed_failed_execution(run: Run) -> str:
    """One failed on-disk attempt + matching ``_ops`` history entry."""
    run.materialize()
    exec_id = f"exec-{run.id}"
    (Path(str(run.run_dir)) / "executions" / exec_id).mkdir(parents=True)
    run.update_ops(
        lambda s: s.model_copy(
            update={
                "status": RunStatus.FAILED,
                "executions": (
                    ExecutionRecord(
                        execution_id=exec_id,
                        started_at=datetime(2026, 7, 1, 10, 0),
                        finished_at=datetime(2026, 7, 1, 10, 5),
                        status="failed",
                    ),
                ),
            }
        )
    )
    return exec_id


async def _reject(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="tester",
        decided_at=datetime(2026, 7, 3, tzinfo=UTC),
        reason="no",
    )


def _lifecycle_proposal(capability_id: str, references: dict[str, str]) -> ChangeProposal:
    proposal = curation_invocation_to_proposal(capability_id, references)
    assert proposal is not None, f"{capability_id} must map to a ChangeProposal"
    return proposal


class TestLifecycleMapping:
    @pytest.mark.parametrize("capability_id", sorted(LIFECYCLE_TABLE))
    def test_maps_to_a_run_lifecycle_proposal(self, capability_id: str) -> None:
        verb, reversibility = LIFECYCLE_TABLE[capability_id]
        proposal = _lifecycle_proposal(capability_id, {"run": "r1"})
        assert proposal.proposed_change.op == "run_lifecycle"
        assert proposal.proposed_change.payload["lifecycle_verb"] == verb
        assert proposal.reversibility == reversibility

    @pytest.mark.parametrize(("literal", "expected"), [("true", True), ("false", False)])
    def test_rerun_fresh_string_is_coerced_to_a_real_bool(
        self, literal: str, expected: bool
    ) -> None:
        proposal = _lifecycle_proposal(
            "molexp.lifecycle.run_rerun", {"run": "r1", "fresh": literal}
        )
        assert bool(proposal.proposed_change.payload.get("fresh", False)) is expected

    def test_prune_statuses_reference_is_split_to_a_list(self) -> None:
        proposal = _lifecycle_proposal(
            "molexp.lifecycle.runs_prune", {"run": "r1", "statuses": "failed,cancelled"}
        )
        assert proposal.proposed_change.payload["statuses"] == ["failed", "cancelled"]

    def test_no_known_destructive_cap_maps_to_none(self) -> None:
        """The CurationArgumentError escape is unreachable for known caps."""
        cases: dict[str, dict[str, str]] = {
            "molexp.curation.move_run": {"run": "r1", "target_experiment": "e2"},
            "molexp.curation.delete_folder": {"folder": "r1"},
            "molexp.curation.rehome_asset": {
                "asset": "a1",
                "source": "experiment:e1",
                "target": "project:p",
            },
            **{cap_id: {"run": "r1"} for cap_id in LIFECYCLE_TABLE},
        }
        for cap_id, references in cases.items():
            assert curation_invocation_to_proposal(cap_id, references) is not None, cap_id


class TestRehomeColonRefs:
    def test_colon_refs_build_the_rehome_proposal(self) -> None:
        proposal = curation_invocation_to_proposal(
            "molexp.curation.rehome_asset",
            {"asset": "a1", "source": "experiment:exp-a", "target": "project:demo"},
        )
        assert proposal is not None
        assert proposal.proposed_change.op == "asset_move"
        payload = proposal.proposed_change.payload
        assert payload["curation_op"] == "rehome_asset"
        assert payload["source"] == {"kind": "experiment", "id": "exp-a"}
        assert payload["target"] == {"kind": "project", "id": "demo"}

    def test_action_defaults_to_copy_and_stays_reversible(self) -> None:
        proposal = curation_invocation_to_proposal(
            "molexp.curation.rehome_asset",
            {"asset": "a1", "source": "experiment:exp-a", "target": "project:demo"},
        )
        assert proposal is not None
        assert proposal.proposed_change.payload["action"] == "copy"
        assert proposal.reversibility == "reversible"

    def test_move_action_is_partially_reversible(self) -> None:
        proposal = curation_invocation_to_proposal(
            "molexp.curation.rehome_asset",
            {
                "asset": "a1",
                "source": "experiment:exp-a",
                "target": "project:demo",
                "action": "move",
            },
        )
        assert proposal is not None
        assert proposal.reversibility == "partially"

    def test_scope_ref_without_a_colon_fails_loudly(self) -> None:
        with pytest.raises(ValueError):
            curation_invocation_to_proposal(
                "molexp.curation.rehome_asset",
                {"asset": "a1", "source": "exp-a", "target": "project:demo"},
            )


class TestDeniedProposalMutatesNothing:
    def test_denied_prune_leaves_dirs_and_history_intact(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        target = _target_run(ws)
        exec_id = _seed_failed_execution(target)
        audit = _audit_run(ws)
        proposal = _lifecycle_proposal(
            "molexp.lifecycle.runs_prune", {"run": "r1", "statuses": "failed"}
        )

        result = asyncio.run(
            run_curation_proposal(proposal, workspace=ws, run=audit, approve=_reject)
        )

        assert result.execution_result is not None
        assert result.execution_result.status == "rejected"
        assert (Path(str(target.run_dir)) / "executions" / exec_id).exists()
        fresh = Workspace(root=tmp_path, name="Lab")
        reloaded = fresh.get_project("p").get_experiment("e1").get_run("r1")
        assert [rec.execution_id for rec in reloaded.execution_history] == [exec_id]
        assert reloaded.status == "failed"
