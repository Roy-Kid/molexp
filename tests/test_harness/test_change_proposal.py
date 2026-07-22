"""High-risk-mutation classifier + the dry, audited ``ChangeProposal`` gate.

Mirrors :mod:`molexp.harness.change_proposal_gate` (the §8.1 classifier and the
store-gate-audit chokepoint). The frozen schemas themselves live in
:mod:`molexp.harness.schemas.change_proposal`.

References:
- spec:       ``.claude/specs/change-proposal.md``
- acceptance: ``.claude/specs/change-proposal.acceptance.md``
- design:     ``.claude/notes/integration.md`` §6, §8
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from molexp.harness.change_proposal_gate import (
    approval_level_for,
    gate_change_proposal,
    is_high_risk,
    requires_change_proposal,
)
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.schemas.change_proposal import (
    CHANGE_PROPOSAL_KIND,
    ChangeProposal,
    ChangeSpec,
    ObjectRef,
    StateSnapshot,
)
from molexp.harness.stages.approval_gate import auto_grant_approver
from molexp.workspace import SourceRef

_NOW = datetime(2026, 7, 1, tzinfo=UTC)

HIGH_RISK_OPS = (
    "workflow_change",
    "param_space_change",
    "artifact_delete",
    "knowledge_change",
    "run_lifecycle",
    "asset_move",
    "api_schema_change",
)


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
        run_id="run-cp",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
    )


def _proposal(op: str = "workflow_change", reversibility: str = "reversible") -> ChangeProposal:
    return ChangeProposal(
        id="cp-1",
        intent="rewire the build workflow",
        current_state=StateSnapshot(objects=[ObjectRef(kind="workflow", id="wf-1")]),
        proposed_change=ChangeSpec(op=op, summary="swap task A for B", payload={"diff": "..."}),
        affected_objects=[ObjectRef(kind="workflow", id="wf-1")],
        expected_benefit="faster build",
        risks=["breaks downstream"],
        reversibility=reversibility,
        approval_level=approval_level_for(op, reversibility),
        evidence=[],
        knowledge=[SourceRef(kind="run", ref="r1")],
    )


class TestHighRiskClassifier:
    def test_every_high_risk_op_requires_a_proposal(self) -> None:
        """ac-002 — every §8.1 op is high-risk; a read-only op is not."""
        for op in HIGH_RISK_OPS:
            assert is_high_risk(op) is True
            assert requires_change_proposal(op) is True
        assert is_high_risk("read_summary") is False
        assert requires_change_proposal("read_summary") is False

    def test_irreversible_change_escalates_approval_level(self) -> None:
        """ac-002 — irreversible forces at least ``elevated``; otherwise ``user``."""
        assert approval_level_for("workflow_change", "irreversible") == "elevated"
        assert approval_level_for("workflow_change", "reversible") == "user"
        assert approval_level_for("run_lifecycle", "partially") == "user"


class TestGateChangeProposal:
    def test_granted_records_request_and_stores_artifact(self, tmp_path: Path) -> None:
        """ac-003 — created → gated (granted) → audited, dry."""
        ctx = _ctx(tmp_path)
        result = asyncio.run(
            gate_change_proposal(ctx, _proposal(), approve=auto_grant_approver, now=_NOW)
        )
        assert result.execution_result is not None
        assert result.execution_result.status == "granted"

        types = [ev.type for ev in ctx.event_log.list_events("run-cp")]
        assert "approval_requested" in types
        assert "approval_granted" in types

        ref_id = result.execution_result.result_artifact_ids[0]
        assert ctx.artifact_store.get_ref(ref_id).kind == CHANGE_PROPOSAL_KIND

    def test_rejected_is_recorded_and_preserved_not_aborted(self, tmp_path: Path) -> None:
        """ac-004 — a rejected proposal is recorded (§8.3), not aborted, and kept."""
        ctx = _ctx(tmp_path)

        async def _reject(request: ApprovalRequest) -> ApprovalDecision:
            return ApprovalDecision(
                request_id=request.id,
                granted=False,
                decided_by="reviewer",
                decided_at=_NOW,
                reason="too risky",
            )

        result = asyncio.run(gate_change_proposal(ctx, _proposal(), approve=_reject, now=_NOW))
        assert result.execution_result is not None
        assert result.execution_result.status == "rejected"

        types = [ev.type for ev in ctx.event_log.list_events("run-cp")]
        assert "approval_rejected" in types

        # the rejected proposal is still stored (preserved, §8.3)
        ref_id = result.execution_result.result_artifact_ids[0]
        assert ctx.artifact_store.get_ref(ref_id).kind == CHANGE_PROPOSAL_KIND
