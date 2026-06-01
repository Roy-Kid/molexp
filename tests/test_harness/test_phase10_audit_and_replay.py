"""Phase-10 tests: AuditReport + generate_audit_report + replay/resume."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


@pytest.fixture()
def stores(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    return a, e, p


# ============================================================ AuditReport


def test_audit_report_frozen_and_defaults() -> None:
    from molexp.harness.schemas.audit_report import AuditReport

    r = AuditReport(run_id="r", summary="s")
    assert r.root_artifact_id is None
    assert r.final_artifact_ids == []
    assert r.approvals == []
    assert r.validation_results == []
    assert r.failures == []
    assert r.command_summaries == []
    assert r.limitations == []
    with pytest.raises(ValidationError):
        r.summary = "mutated"  # type: ignore[misc]


def test_audit_report_round_trip() -> None:
    from molexp.harness.schemas.audit_report import AuditReport

    r = AuditReport(
        run_id="r",
        summary="s",
        approvals=[{"intent": "hpc_submission"}],
        validation_results=["v1", "v2"],
        failures=[{"stage": "x", "error": "y"}],
    )
    r2 = AuditReport.model_validate_json(r.model_dump_json())
    assert r2 == r


# ============================================================ generate_audit_report


def test_generate_audit_report_assembles_known_facts(stores) -> None:
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

    report = generate_audit_report(run_id=run_id, event_log=e, artifact_store=a, provenance_store=p)
    assert report.run_id == run_id
    assert any(d.get("artifact_ids") == ["req-x"] for d in report.approvals)
    assert val.id in report.validation_results
    assert any(f["stage"] == "FailingStage" for f in report.failures)
    assert report.root_artifact_id == up.id


# ============================================================ replay_metadata


def test_replay_metadata_returns_events_in_seq_order(stores) -> None:
    from molexp.harness.audit.replay import replay_metadata

    _a, e, _p = stores
    for i in range(3):
        e.append(run_id="rx", type="stage_started", actor="harness", payload={"i": i})
    events = replay_metadata(e, "rx")
    assert [ev.seq for ev in events] == [1, 2, 3]


# ============================================================ find_last_successful_stage


def test_find_last_empty_run(stores) -> None:
    from molexp.harness.audit.replay import find_last_successful_stage

    _a, e, _p = stores
    assert find_last_successful_stage(e, "empty") is None


def test_find_last_only_started(stores) -> None:
    from molexp.harness.audit.replay import find_last_successful_stage

    _a, e, _p = stores
    e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
    assert find_last_successful_stage(e, "r") is None


def test_find_last_completed_no_failure(stores) -> None:
    from molexp.harness.audit.replay import find_last_successful_stage

    _a, e, _p = stores
    e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
    e.append(run_id="r", type="stage_completed", actor="harness", payload={"stage": "A"})
    assert find_last_successful_stage(e, "r") == "A"


def test_find_last_failure_of_different_stage_preserves_prior_completion(stores) -> None:
    """A failed AFTER B completed should NOT invalidate B (different stage). Resume from B."""
    from molexp.harness.audit.replay import find_last_successful_stage

    _a, e, _p = stores
    e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
    e.append(run_id="r", type="stage_completed", actor="harness", payload={"stage": "A"})
    e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "B"})
    e.append(run_id="r", type="stage_failed", actor="harness", payload={"stage": "B"})
    # New semantics: A's completion is not invalidated by B's failure.
    assert find_last_successful_stage(e, "r") == "A"


def test_find_last_failure_of_same_stage_invalidates_its_completion(stores) -> None:
    """Stage completed then rerun and failed → its completion is invalidated."""
    from molexp.harness.audit.replay import find_last_successful_stage

    _a, e, _p = stores
    e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
    e.append(run_id="r", type="stage_completed", actor="harness", payload={"stage": "A"})
    e.append(run_id="r", type="stage_started", actor="harness", payload={"stage": "A"})
    e.append(run_id="r", type="stage_failed", actor="harness", payload={"stage": "A"})
    assert find_last_successful_stage(e, "r") is None


# ============================================================ surface


def test_phase10_public_surface() -> None:
    from molexp.harness import (  # noqa: F401
        AuditReport,
        find_last_successful_stage,
        generate_audit_report,
        replay_metadata,
    )
