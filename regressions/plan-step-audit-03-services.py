"""Regression: decide_plan_review approve via public plan_runtime API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from molexp.harness.schemas import ApprovalRequest, ReviewDecision
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.services.plan_runtime import (
    build_review_pack,
    decide_plan_review,
    render_approval_preview,
)
from molexp.workspace import Workspace


def main() -> int:
    with TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp) / "lab", name="lab")
        ws.materialize()
        run = ws.add_project("p").add_experiment("e").add_run(id="r1")
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        FileArtifactStore(root=Path(str(run.run_dir)) / "artifacts").put_json(
            kind="experiment_spec",
            obj={"title": "T", "objective": "O"},
            created_by="reg",
            parent_ids=[],
        )
        pack = build_review_pack(run, "experiment_spec")
        assert pack.pack_id
        assert render_approval_preview(run, "experiment_spec")

        request = ApprovalRequest(
            id="req-reg-03",
            intent="experiment_spec",
            reason="reg",
            triggered_by_policy="reg",
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
        )
        decision = ReviewDecision(
            pack_id=pack.pack_id,
            action="approve",
            decided_by="regression",
            decided_at=datetime(2026, 7, 19, 12, tzinfo=UTC),
        )
        approval = decide_plan_review(run=run, request=request, decision=decision)
        assert approval.granted is True
        store = SQLiteApprovalStore(path=Path(str(run.run_dir)) / "harness.sqlite")
        assert store.granted_decision_for("req-reg-03") is not None
        print("plan-step-audit-03-services: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
