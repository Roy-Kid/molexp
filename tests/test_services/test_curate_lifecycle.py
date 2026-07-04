"""NL curation flow x run-lifecycle verbs + rehome colon refs (vision-loop-07 §3/§4).

Two contracts of the ``services.curate_runtime`` proposal mapping:

* **The five ``molexp.lifecycle.*`` ids map onto ChangeProposals** (op
  ``run_lifecycle``, verb discriminated by ``payload["lifecycle_verb"]`` —
  the ``asset_move``/``curation_op`` precedent), with the spec's reversibility
  table: ``irreversible`` for prune, ``partially`` for the execute family,
  ``reversible`` for cancel. No known destructive capability maps to ``None``
  any more — the ``CurationArgumentError`` escape is unreachable for them.
* **``rehome_asset`` is now expressible in the flat reference map** via
  colon-encoded scope refs (``source: "experiment:exp-a"``,
  ``target: "project:demo"``).

Gating end-to-end (spec Testing strategy): a proposal denied by the approver
mutates NOTHING — run status, ``_ops`` history, and disk are untouched; the
outcome records ``rejected`` and never raises through the flow. Idioms mirror
``tests/test_server/test_curate_proposal_flow.py``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.change_proposal_gate import approval_level_for
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, ChangeProposal
from molexp.harness.schemas.change_proposal import ObjectRef
from molexp.services.curate_runtime import (
    curation_invocation_to_proposal,
    run_curation_proposal,
)
from molexp.workspace import Workspace
from molexp.workspace.models import ExecutionRecord, RunStatus
from molexp.workspace.run import Run

# ── shared fixtures / helpers ────────────────────────────────────────────────

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


# ── mapping: the five lifecycle ids ──────────────────────────────────────────


class TestLifecycleMapping:
    @pytest.mark.parametrize("capability_id", sorted(LIFECYCLE_TABLE))
    def test_maps_to_a_run_lifecycle_proposal(self, capability_id: str) -> None:
        verb, reversibility = LIFECYCLE_TABLE[capability_id]
        proposal = _lifecycle_proposal(capability_id, {"run": "r1"})
        assert proposal.proposed_change.op == "run_lifecycle"
        assert proposal.proposed_change.payload["lifecycle_verb"] == verb
        assert proposal.reversibility == reversibility

    @pytest.mark.parametrize("capability_id", sorted(LIFECYCLE_TABLE))
    def test_affects_exactly_the_named_run(self, capability_id: str) -> None:
        proposal = _lifecycle_proposal(capability_id, {"run": "r1"})
        assert ObjectRef(kind="run", id="r1") in proposal.affected_objects
        run_refs = [o for o in proposal.affected_objects if o.kind == "run"]
        assert len(run_refs) == 1

    @pytest.mark.parametrize("capability_id", sorted(LIFECYCLE_TABLE))
    def test_approval_level_follows_reversibility(self, capability_id: str) -> None:
        _, reversibility = LIFECYCLE_TABLE[capability_id]
        proposal = _lifecycle_proposal(capability_id, {"run": "r1"})
        assert proposal.approval_level == approval_level_for("run_lifecycle", reversibility)

    def test_prune_escalates_to_elevated(self) -> None:
        proposal = _lifecycle_proposal("molexp.lifecycle.runs_prune", {"run": "r1"})
        assert proposal.approval_level == "elevated"

    def test_rerun_fresh_true_is_parsed_to_a_real_bool(self) -> None:
        proposal = _lifecycle_proposal("molexp.lifecycle.run_rerun", {"run": "r1", "fresh": "true"})
        assert proposal.proposed_change.payload.get("fresh") is True

    def test_rerun_fresh_false_never_becomes_truthy(self) -> None:
        """A "false" string reference must not survive as a truthy payload value."""
        proposal = _lifecycle_proposal(
            "molexp.lifecycle.run_rerun", {"run": "r1", "fresh": "false"}
        )
        assert bool(proposal.proposed_change.payload.get("fresh", False)) is False

    def test_prune_statuses_reference_is_a_comma_list(self) -> None:
        proposal = _lifecycle_proposal(
            "molexp.lifecycle.runs_prune", {"run": "r1", "statuses": "failed,cancelled"}
        )
        assert proposal.proposed_change.payload["statuses"] == ["failed", "cancelled"]

    def test_prune_execution_ids_reference_is_a_comma_list(self) -> None:
        proposal = _lifecycle_proposal(
            "molexp.lifecycle.runs_prune",
            {"run": "r1", "execution_ids": "exec-r1,exec-r1-2"},
        )
        assert proposal.proposed_change.payload["execution_ids"] == ["exec-r1", "exec-r1-2"]

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


# ── mapping: rehome_asset colon-encoded scope refs ───────────────────────────


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


# ── gating end-to-end: a denied proposal mutates nothing ─────────────────────


class TestDeniedProposalMutatesNothing:
    def test_denied_run_execute_leaves_the_run_pending(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        target = _target_run(ws)
        target.materialize()
        audit = _audit_run(ws)
        proposal = _lifecycle_proposal("molexp.lifecycle.run_execute", {"run": "r1"})

        result = asyncio.run(
            run_curation_proposal(proposal, workspace=ws, run=audit, approve=_reject)
        )

        assert result.execution_result is not None
        assert result.execution_result.status == "rejected"
        fresh = Workspace(root=tmp_path, name="Lab")
        reloaded = fresh.get_project("p").get_experiment("e1").get_run("r1")
        assert reloaded.status == "pending"
        assert reloaded.execution_history == []
        assert not (Path(str(reloaded.run_dir)) / "executions").exists()

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

    def test_denied_cancel_leaves_a_running_run_running(self, tmp_path: Path) -> None:
        import os
        import platform

        from molexp.workspace.run_ops import RunOpsState

        ws = _workspace(tmp_path)
        target = _target_run(ws)
        target.materialize()
        target.write_ops(
            RunOpsState(
                status="running",
                owner_pid=os.getpid(),
                owner_host=platform.node(),
                heartbeat_at=datetime.now(UTC),
            )
        )
        audit = _audit_run(ws)
        proposal = _lifecycle_proposal("molexp.lifecycle.run_cancel", {"run": "r1"})

        result = asyncio.run(
            run_curation_proposal(proposal, workspace=ws, run=audit, approve=_reject)
        )

        assert result.execution_result is not None
        assert result.execution_result.status == "rejected"
        assert _target_run(Workspace(root=tmp_path, name="Lab")).status == "running"
