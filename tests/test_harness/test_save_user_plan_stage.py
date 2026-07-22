"""Tests for the ``SaveUserPlan`` stage (``molexp.harness.stages.save_user_plan``)."""

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


class TestSaveUserPlan:
    def test_writes_raw_and_structured_artifacts_with_provenance_edge(self, ctx) -> None:
        """Raw text + structured JSON, both ``user_plan`` kind, raw → structured edge."""
        from molexp.harness.core.stage_runner import StageRunner
        from molexp.harness.stages.save_user_plan import SaveUserPlan

        runner = StageRunner(ctx)
        structured_ref = asyncio.run(
            runner.run_stage(SaveUserPlan(user_text="Simulate water at 300K"))
        )

        refs = ctx.artifact_store.list_by_kind("user_plan")
        assert len(refs) == 2

        # Returned ref is the structured JSON; the other is the raw text.
        raw_refs = [r for r in refs if r.id != structured_ref.id]
        assert len(raw_refs) == 1
        raw_ref = raw_refs[0]

        # StageRunner wires the derived_from edge from parent_ids.
        assert raw_ref.id in structured_ref.parent_ids
        ancestors = ctx.lineage_store.trace_backward(structured_ref.id)
        assert [r.id for r in ancestors] == [raw_ref.id]

    def test_structured_artifact_is_valid_user_plan_json(self, ctx) -> None:
        """The returned structured artifact is a valid ``UserPlan`` envelope."""
        from molexp.harness.core.stage_runner import StageRunner
        from molexp.harness.schemas.user_plan import UserPlan
        from molexp.harness.stages.save_user_plan import SaveUserPlan

        runner = StageRunner(ctx)
        structured_ref = asyncio.run(
            runner.run_stage(SaveUserPlan(user_text="simulate water", user_id="alice"))
        )

        plan = UserPlan.model_validate(json.loads(ctx.artifact_store.get(structured_ref.id)))
        assert plan.raw_text == "simulate water"
        assert plan.user_id == "alice"
