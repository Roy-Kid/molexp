"""Regression: StepAuditLoop hard suspend then grant via public harness API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import ApprovalPendingError
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, FormDocument, ReviewPack
from molexp.harness.schemas.step_audit import FormTextField
from molexp.harness.stages import StepAuditLoop
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore


def _pack(_ctx: HarnessRunContext) -> ReviewPack:
    return ReviewPack(
        pack_id="pack-reg-02",
        step_id="draft_spec",
        step_title="Draft spec",
        policy="hard",
        summary_md="reg",
        form=FormDocument(fields=[FormTextField(id="n", label="N")]),
        decision_options=["approve", "reject"],
    )


async def _run() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = root / "harness.sqlite"
        artifacts = FileArtifactStore(root=root / "artifacts")
        event_log = SQLiteEventLog(path=db)
        store = SQLiteApprovalStore(db)
        ctx = HarnessRunContext(
            run_id="run-reg-02",
            workspace_root=root,
            artifact_store=artifacts,
            event_log=event_log,
            lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=artifacts),
            approval_store=store,
        )
        artifacts.put_json(
            kind="experiment_spec",
            obj={"id": "s", "title": "T", "objective": "O", "experiment_report_id": "r"},
            created_by="reg",
            parent_ids=[],
        )
        request = ApprovalRequest(
            id="req-reg-02",
            intent="experiment_spec",
            reason="reg",
            triggered_by_policy="reg",
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
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
        try:
            await loop.run(ctx)
            raise AssertionError("expected ApprovalPendingError")
        except ApprovalPendingError as pending:
            assert pending.run_id == "run-reg-02"
            assert [r.id for r in pending.requests] == ["req-reg-02"]

        store.record_decision(
            ApprovalDecision(
                request_id="req-reg-02",
                granted=True,
                decided_by="regression",
                decided_at=datetime(2026, 7, 19, 12, tzinfo=UTC),
                reason="granted",
            )
        )
        summary = await loop.run(ctx)
        assert summary.kind == "spec_approval"
        print("plan-step-audit-02-loop: ok")


def main() -> int:
    import asyncio

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
