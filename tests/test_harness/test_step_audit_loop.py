"""Unit tests for StepAuditLoop auto / soft / hard policy matrix."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import ApprovalPendingError, StageExecutionError
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, FormDocument, ReviewPack
from molexp.harness.schemas.step_audit import FormTextField
from molexp.harness.stages import ApprovalGate, StepAuditLoop, auto_grant_approver
from molexp.harness.stages.review_pack_builders import (
    build_experiment_spec_review_pack,
    build_plan_review_pack,
)
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

_RUN_ID = "run-audit"


def _request(request_id: str = "req-audit-1") -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id,
        intent="experiment_spec",
        reason="test gate",
        triggered_by_policy="test",
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )


def _pack(_ctx: HarnessRunContext) -> ReviewPack:
    return ReviewPack(
        pack_id="pack-test",
        step_id="draft_spec",
        step_title="Draft spec",
        policy="hard",
        summary_md="test pack",
        form=FormDocument(fields=[FormTextField(id="n", label="N")]),
        decision_options=["approve", "reject", "revise"],
    )


class _Env:
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
        self.subject = self.artifacts.put_json(
            kind="experiment_spec",
            obj={"id": "spec-1", "title": "T", "objective": "O", "experiment_report_id": "r"},
            created_by="test",
            parent_ids=[],
        )


@pytest.fixture()
def env(tmp_path: Path) -> _Env:
    return _Env(tmp_path)


class TestStepAuditPolicies:
    async def test_auto_approves_without_pending(self, env: _Env) -> None:
        loop = StepAuditLoop(
            name="approve_experiment_spec",
            subject_kind="experiment_spec",
            pack_builder=_pack,
            policy="auto",
            request=_request(),
            result_kind="spec_approval",
        )
        ref = await loop.run(env.ctx)
        assert ref.kind == "spec_approval"
        assert env.approval_store.pending(_RUN_ID) == []
        assert env.artifacts.latest_by_kind("review_pack") is not None
        types = [e.type for e in env.event_log.list_events(_RUN_ID)]
        assert "approval_requested" in types
        assert "approval_granted" in types

    async def test_soft_persists_pack_without_pending(self, env: _Env) -> None:
        loop = StepAuditLoop(
            name="soft_audit",
            subject_kind="experiment_spec",
            pack_builder=_pack,
            policy="soft",
            request=_request(),
        )
        ref = await loop.run(env.ctx)
        assert ref.kind == "review_pack"
        assert env.approval_store.pending(_RUN_ID) == []

    async def test_hard_suspends_when_no_approver(self, env: _Env) -> None:
        loop = StepAuditLoop(
            name="approve_experiment_spec",
            subject_kind="experiment_spec",
            pack_builder=_pack,
            policy="hard",
            request=_request(),
            approve=None,
            result_kind="spec_approval",
        )
        with pytest.raises(ApprovalPendingError):
            await loop.run(env.ctx)
        assert env.artifacts.latest_by_kind("review_pack") is not None
        [pending] = env.approval_store.pending(_RUN_ID)
        assert pending.id == "req-audit-1"

    async def test_hard_grant_via_live_approver(self, env: _Env) -> None:
        loop = StepAuditLoop(
            name="approve_experiment_spec",
            subject_kind="experiment_spec",
            pack_builder=_pack,
            policy="hard",
            request=_request(),
            approve=auto_grant_approver,
            result_kind="spec_approval",
        )
        ref = await loop.run(env.ctx)
        assert ref.kind == "spec_approval"
        grant = env.approval_store.granted_decision_for("req-audit-1")
        assert grant is not None and grant.granted is True

    async def test_hard_grant_after_stored_decision(self, env: _Env) -> None:
        request = _request()
        env.approval_store.record_pending(_RUN_ID, request)
        env.approval_store.record_decision(
            ApprovalDecision(
                request_id=request.id,
                granted=True,
                decided_by="ui-operator",
                decided_at=datetime(2026, 7, 19, 12, tzinfo=UTC),
                reason="ok",
            )
        )
        loop = StepAuditLoop(
            name="approve_experiment_spec",
            subject_kind="experiment_spec",
            pack_builder=_pack,
            policy="hard",
            request=request,
            approve=None,
            result_kind="spec_approval",
        )
        ref = await loop.run(env.ctx)
        assert ref.kind == "spec_approval"

    async def test_missing_subject_raises(self, env: _Env) -> None:
        loop = StepAuditLoop(
            name="x",
            subject_kind="workflow_source",
            pack_builder=_pack,
            policy="auto",
            request=_request(),
        )
        with pytest.raises(StageExecutionError, match="missing subject"):
            await loop.run(env.ctx)


class TestPackBuilders:
    def test_experiment_spec_builder(self, env: _Env) -> None:
        pack = build_experiment_spec_review_pack(env.ctx)
        assert pack.step_id == "draft_spec"
        assert pack.policy == "hard"
        assert pack.subject_refs

    def test_plan_builder_prefers_execution_result(self, env: _Env) -> None:
        env.artifacts.put_json(
            kind="workflow_source",
            obj={"module_name": "wf", "source": "pass"},
            created_by="test",
            parent_ids=[],
        )
        env.artifacts.put_json(
            kind="execution_result",
            obj={"status": "succeeded", "metadata": {"mode": "compile"}},
            created_by="test",
            parent_ids=[],
        )
        pack = build_plan_review_pack(env.ctx)
        assert pack.step_id == "review"
        assert any(r.kind == "execution_result" for r in pack.subject_refs)


class TestPlanModeWiring:
    def test_step2_and_step8_are_step_audit_loops(self) -> None:
        from molexp.harness import PlanMode

        stages = {s.name: s for s in PlanMode().stages("draft")}
        assert isinstance(stages["approve_experiment_spec"], StepAuditLoop)
        assert isinstance(stages["approve_plan"], StepAuditLoop)
        # Policy is constructor-only; name + type pin the PlanMode migration.
        assert stages["approve_experiment_spec"].name == "approve_experiment_spec"
        assert stages["approve_plan"].name == "approve_plan"

    def test_execute_tail_still_uses_approval_gate(self) -> None:
        from molexp.harness import PlanMode

        stages = {s.name: s for s in PlanMode(execute=True).stages("draft")}
        assert isinstance(stages["approve_execution"], ApprovalGate)
        assert not isinstance(stages["approve_execution"], StepAuditLoop)

    def test_nine_step_name_sequence_unchanged(self) -> None:
        from molexp.harness import PlanMode

        names = [s.name for s in PlanMode().stages("draft")]
        assert names == [
            "save_user_plan",
            "assemble_knowledge_context",
            "generate_experiment_report",
            "generate_experiment_spec",
            "approve_experiment_spec",
            "resolve_capabilities",
            "extract_workflow_ir",
            "bind_molcrafts_tasks",
            "generate_workflow_source",
            "generate_test_spec",
            "generate_test_code",
            "generate_input_set",
            "materialize_execution",
            "execute_tests",
            "compile_workflow",
            "approve_plan",
            "generate_execution_report",
        ]
