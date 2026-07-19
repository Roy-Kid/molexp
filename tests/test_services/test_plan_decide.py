"""Tests for decide_plan_review three-action path."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from molexp.harness.schemas import ApprovalRequest, ReviewDecision
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.services.plan_runtime.decide import decide_plan_review
from molexp.workspace import Workspace


def _run(tmp_path: Path):
    ws = Workspace(tmp_path / "lab", name="lab")
    ws.materialize()
    return ws.add_project("p").add_experiment("e").add_run(id="r1")


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-1",
        intent="experiment_spec",
        reason="test",
        triggered_by_policy="test",
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )


def _decision(action: str) -> ReviewDecision:
    return ReviewDecision(
        pack_id="pack-1",
        action=action,  # type: ignore[arg-type]
        decided_by="tester",
        decided_at=datetime(2026, 7, 19, 12, tzinfo=UTC),
        reason="because",
        field_values={"n": "v"} if action == "revise" else {},
    )


def test_approve_grants_and_resumes(tmp_path: Path) -> None:
    run = _run(tmp_path)
    task = MagicMock()
    task.status = "waiting_approval"
    approval = decide_plan_review(
        run=run,
        request=_request(),
        decision=_decision("approve"),
        task=task,
    )
    assert approval.granted is True
    store = SQLiteApprovalStore(path=Path(str(run.run_dir)) / "harness.sqlite")
    assert store.granted_decision_for("req-1") is not None
    task.resume.assert_called_once()
    task.mark_rejected.assert_not_called()


def test_reject_marks_rejected(tmp_path: Path) -> None:
    run = _run(tmp_path)
    task = MagicMock()
    task.status = "waiting_approval"
    approval = decide_plan_review(
        run=run,
        request=_request(),
        decision=_decision("reject"),
        task=task,
    )
    assert approval.granted is False
    task.mark_rejected.assert_called_once()
    task.resume.assert_not_called()


def test_revise_resumes_without_mark_rejected(tmp_path: Path) -> None:
    run = _run(tmp_path)
    task = MagicMock()
    task.status = "waiting_approval"
    approval = decide_plan_review(
        run=run,
        request=_request(),
        decision=_decision("revise"),
        task=task,
    )
    assert approval.granted is False
    task.resume.assert_called_once()
    task.mark_rejected.assert_not_called()
    # decision artifact persisted
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(str(run.run_dir)) / "artifacts")
    ref = store.latest_by_kind("review_decision")
    assert ref is not None
    body = ReviewDecision.model_validate_json(store.get(ref.id))
    assert body.action == "revise"
    assert body.field_values == {"n": "v"}


def test_invalid_action_raises(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(ValidationError):
        decide_plan_review(
            run=run,
            request=_request(),
            decision={
                "pack_id": "p",
                "action": "defer",
                "decided_by": "x",
                "decided_at": datetime.now(tz=UTC).isoformat(),
            },
        )
