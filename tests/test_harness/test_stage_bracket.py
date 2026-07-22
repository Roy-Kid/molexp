"""Audit-bracket tests for :func:`run_stage_bracketed` + :class:`StageRunner`.

The bracket is the single execution path for a harness stage — used directly
by ``Mode`` and via the thin ``StageRunner`` wrapper. Contract:

- success path → ``stage_started`` / ``artifact_created`` / ``stage_completed``
- ``StagePersistedFailureError`` → ``stage_started`` / ``artifact_created``
  / ``stage_failed`` plus the persisted ref's ``derived_from`` edges
- plain exception → ``stage_started`` / ``stage_failed`` wrapped in
  ``StageExecutionError``
- every ``derived_from`` edge is stamped with the producing stage + run id
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.core.stage_runner import StageRunner, run_stage_bracketed
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import PlanArtifactRef
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

# ───────────────────────────────────────────────────────── fixtures / helpers


def _make_ctx(root: Path, *, run_id: str = "run-test") -> HarnessRunContext:
    """Build a fresh HarnessRunContext backed by isolated on-disk stores."""
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id=run_id,
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=provenance,
    )


class SeedStage(Stage):
    name = "SeedStage"

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        return ctx.artifact_store.put_json(
            kind="user_plan",
            obj={"step": "A"},
            created_by="SeedStage",
            parent_ids=[],
        )


class ChildStage(Stage):
    name = "ChildStage"

    def __init__(self, parent_id: str) -> None:
        self._parent_id = parent_id

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        return ctx.artifact_store.put_json(
            kind="experiment_report",
            obj={"step": "B"},
            created_by="ChildStage",
            parent_ids=[self._parent_id],
        )


class PlainFailStage(Stage):
    name = "PlainFailStage"

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        raise RuntimeError("boom")


def _persist_then_raise_stage(parent_id: str) -> type[Stage]:
    class PersistThenRaiseStage(Stage):
        name = "PersistThenRaiseStage"

        async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
            failure_ref = ctx.artifact_store.put_json(
                kind="validation_report",
                obj={"passed": False, "violations": []},
                created_by="PersistThenRaiseStage",
                parent_ids=[parent_id],
            )
            raise StagePersistedFailureError(failure_ref, "parse failed")

    return PersistThenRaiseStage


class TestRunStageBracketed:
    def test_success_emits_started_created_completed(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        ref = asyncio.run(run_stage_bracketed(ctx, SeedStage()))

        events = ctx.event_log.list_events("run-test")
        assert [e.type for e in events] == [
            "stage_started",
            "artifact_created",
            "stage_completed",
        ]
        assert ref.id in events[1].artifact_ids

    def test_persisted_failure_records_artifact_then_stage_failed(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        parent = ctx.artifact_store.put_json(
            kind="user_plan", obj={"step": "parent"}, created_by="seed", parent_ids=[]
        )
        with pytest.raises(StagePersistedFailureError):
            asyncio.run(run_stage_bracketed(ctx, _persist_then_raise_stage(parent.id)()))

        assert [e.type for e in ctx.event_log.list_events("run-test")] == [
            "stage_started",
            "artifact_created",
            "stage_failed",
        ]
        # The persisted failure report's lineage is recorded before the failure.
        descendants = ctx.lineage_store.trace_forward(parent.id)
        assert any(d.kind == "validation_report" for d in descendants)

    def test_plain_exception_wraps_in_stage_execution_error(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        with pytest.raises(StageExecutionError) as exc:
            asyncio.run(run_stage_bracketed(ctx, PlainFailStage()))

        assert isinstance(exc.value.__cause__, RuntimeError)
        assert [e.type for e in ctx.event_log.list_events("run-test")] == [
            "stage_started",
            "stage_failed",
        ]

    def test_derived_from_edge_is_stamped_with_stage_and_run_id(self, tmp_path: Path) -> None:
        """The bracket wires the returned ref's parent_ids into a stamped edge."""
        ctx = _make_ctx(tmp_path / "ctx", run_id="run-lineage")
        parent = asyncio.run(run_stage_bracketed(ctx, SeedStage()))
        child = asyncio.run(run_stage_bracketed(ctx, ChildStage(parent.id)))

        edges = ctx.lineage_store.lineage_graph(parent.id)["edges"]
        assert edges == [
            {
                "parent_id": parent.id,
                "child_id": child.id,
                "relation": "derived_from",
                "stage": "ChildStage",
                "run_id": "run-lineage",
            }
        ]


class TestStageRunner:
    def test_run_stage_produces_identical_audit_to_the_bracket(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        ref = asyncio.run(StageRunner(ctx).run_stage(SeedStage()))

        events = ctx.event_log.list_events("run-test")
        assert [e.type for e in events] == [
            "stage_started",
            "artifact_created",
            "stage_completed",
        ]
        assert ref.id in events[1].artifact_ids
