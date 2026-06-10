"""Tests for SaveUserPlan stage (Phase 2 §SaveUserPlan).

Locks:
- name == "save_user_plan"
- Two user_plan artifacts: raw text first (via put_text), structured JSON
  second (via put_json) referencing raw via parent_ids
- Returned ref is the structured JSON (downstream stages depend on it)
- Through StageRunner: event log = [stage_started, artifact_created, stage_completed]
  and provenance edge raw → structured wired via parent_ids
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db_path = tmp_path / "events.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-save-user-plan",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=provenance,
    )


def test_save_user_plan_name() -> None:
    from molexp.harness.stages.save_user_plan import SaveUserPlan

    assert SaveUserPlan.name == "save_user_plan"


def test_save_user_plan_writes_two_artifacts_and_wires_provenance(ctx) -> None:
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.stages.save_user_plan import SaveUserPlan

    runner = StageRunner(ctx)
    structured_ref = asyncio.run(runner.run_stage(SaveUserPlan(user_text="Simulate water at 300K")))

    # Two artifacts of kind user_plan exist.
    refs = ctx.artifact_store.list_by_kind("user_plan")
    assert len(refs) == 2

    # Returned ref is the structured JSON; the other is the raw text.
    raw_refs = [r for r in refs if r.id != structured_ref.id]
    assert len(raw_refs) == 1
    raw_ref = raw_refs[0]

    # Structured ref carries raw ref in parent_ids; StageRunner has wired
    # the derived_from edge for us.
    assert raw_ref.id in structured_ref.parent_ids
    ancestors = ctx.lineage_store.trace_backward(structured_ref.id)
    assert [r.id for r in ancestors] == [raw_ref.id]


def test_save_user_plan_event_log_quartet(ctx) -> None:
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.stages.save_user_plan import SaveUserPlan

    runner = StageRunner(ctx)
    asyncio.run(runner.run_stage(SaveUserPlan(user_text="hello")))

    events = ctx.event_log.list_events("run-save-user-plan")
    assert [e.type for e in events] == [
        "stage_started",
        "artifact_created",
        "stage_completed",
    ]
    assert events[0].payload == {"stage": "save_user_plan"}
    assert events[1].payload["kind"] == "user_plan"


def test_save_user_plan_structured_json_round_trips_to_user_plan_schema(ctx) -> None:
    """The structured artifact is a valid UserPlan JSON envelope."""
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.schemas.user_plan import UserPlan
    from molexp.harness.stages.save_user_plan import SaveUserPlan

    runner = StageRunner(ctx)
    structured_ref = asyncio.run(
        runner.run_stage(SaveUserPlan(user_text="simulate water", user_id="alice"))
    )

    raw_bytes = ctx.artifact_store.get(structured_ref.id)
    plan = UserPlan.model_validate(json.loads(raw_bytes))
    assert plan.raw_text == "simulate water"
    assert plan.user_id == "alice"
