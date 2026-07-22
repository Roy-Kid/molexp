"""Audit + replay helpers: ``generate_audit_report`` (the pure assembly
function — the ``GenerateAuditReport`` stage wrapper lives in
``test_generate_audit_report_stage.py``), ``replay_metadata``, and the
``find_last_successful_stage`` resume-point contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def stores(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return a, e, p


class TestGenerateAuditReport:
    def test_assembles_approvals_validations_failures_and_root(self, stores) -> None:
        from molexp.harness.audit.generate import generate_audit_report

        a, e, p = stores
        run_id = "run-001"

        # Seed a user_plan artifact + a validation_report artifact.
        up = a.put_text(kind="user_plan", text="x", created_by="u", parent_ids=[])
        val = a.put_json(
            kind="validation_report", obj={"passed": True}, created_by="v", parent_ids=[up.id]
        )

        # Seed events: artifact_created for up + val; approval_requested; stage_failed.
        e.append(
            run_id=run_id, type="stage_started", actor="harness", payload={"stage": "SaveUserPlan"}
        )
        e.append(
            run_id=run_id,
            type="artifact_created",
            actor="harness",
            payload={"stage": "SaveUserPlan", "kind": "user_plan"},
            artifact_ids=[up.id],
        )
        e.append(
            run_id=run_id,
            type="approval_requested",
            actor="harness",
            payload={"intent": "hpc_submission"},
            artifact_ids=["req-x"],
        )
        e.append(
            run_id=run_id,
            type="artifact_created",
            actor="harness",
            payload={"stage": "ValidateWorkflowIR", "kind": "validation_report"},
            artifact_ids=[val.id],
        )
        e.append(
            run_id=run_id,
            type="stage_failed",
            actor="harness",
            payload={"stage": "FailingStage", "error": "oops"},
        )

        report = generate_audit_report(
            run_id=run_id, event_log=e, artifact_store=a, lineage_store=p
        )
        assert report.run_id == run_id
        assert any(d.get("artifact_ids") == ["req-x"] for d in report.approvals)
        assert val.id in report.validation_results
        assert any(f["stage"] == "FailingStage" for f in report.failures)
        assert report.root_artifact_id == up.id


class TestReplayMetadata:
    def test_returns_events_in_seq_order(self, stores) -> None:
        from molexp.harness.audit.replay import replay_metadata

        _a, e, _p = stores
        for i in range(3):
            e.append(run_id="rx", type="stage_started", actor="harness", payload={"i": i})
        events = replay_metadata(e, "rx")
        assert [ev.seq for ev in events] == [1, 2, 3]


class TestFindLastSuccessfulStage:
    def test_returns_none_when_only_started(self, stores) -> None:
        from molexp.harness.audit.replay import find_last_successful_stage

        _a, e, _p = stores
        e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
        assert find_last_successful_stage(e, "r") is None

    def test_returns_the_completed_stage_when_no_failure(self, stores) -> None:
        from molexp.harness.audit.replay import find_last_successful_stage

        _a, e, _p = stores
        e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
        e.append(run_id="r", type="stage_completed", actor="harness", payload={"stage": "A"})
        assert find_last_successful_stage(e, "r") == "A"

    def test_failure_of_a_different_stage_preserves_prior_completion(self, stores) -> None:
        """A failed AFTER B completed should NOT invalidate B (different stage). Resume from B."""
        from molexp.harness.audit.replay import find_last_successful_stage

        _a, e, _p = stores
        e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
        e.append(run_id="r", type="stage_completed", actor="harness", payload={"stage": "A"})
        e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "B"})
        e.append(run_id="r", type="stage_failed", actor="harness", payload={"stage": "B"})
        # A's completion is not invalidated by B's failure.
        assert find_last_successful_stage(e, "r") == "A"

    def test_failure_of_the_same_stage_invalidates_its_completion(self, stores) -> None:
        """Stage completed then rerun and failed → its completion is invalidated."""
        from molexp.harness.audit.replay import find_last_successful_stage

        _a, e, _p = stores
        e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
        e.append(run_id="r", type="stage_completed", actor="harness", payload={"stage": "A"})
        e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
        e.append(run_id="r", type="stage_failed", actor="harness", payload={"stage": "A"})
        assert find_last_successful_stage(e, "r") is None
