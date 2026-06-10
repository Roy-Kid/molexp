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
from molexp.harness.schemas import ArtifactRef
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


class NoopStage(Stage):
    name = "NoopStage"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        return ctx.artifact_store.put_json(
            kind="log",
            obj={"note": "hello"},
            created_by="NoopStage",
            parent_ids=[],
        )


class SeedStage(Stage):
    name = "SeedStage"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
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

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        return ctx.artifact_store.put_json(
            kind="experiment_report",
            obj={"step": "B"},
            created_by="ChildStage",
            parent_ids=[self._parent_id],
        )


class PlainFailStage(Stage):
    name = "PlainFailStage"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        raise RuntimeError("boom")


def _persist_then_raise_stage(parent_id: str) -> type[Stage]:
    class PersistThenRaiseStage(Stage):
        name = "PersistThenRaiseStage"

        async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
            failure_ref = ctx.artifact_store.put_json(
                kind="validation_report",
                obj={"passed": False, "violations": []},
                created_by="PersistThenRaiseStage",
                parent_ids=[parent_id],
            )
            raise StagePersistedFailureError(failure_ref, "parse failed")

    return PersistThenRaiseStage


# ─────────────────────────────────────────────────────────── success path


def test_bracket_success_event_sequence(tmp_path: Path) -> None:
    """Success → stage_started / artifact_created / stage_completed."""
    ctx = _make_ctx(tmp_path)
    ref = asyncio.run(run_stage_bracketed(ctx, SeedStage()))

    events = ctx.event_log.list_events("run-test")
    assert [e.type for e in events] == ["stage_started", "artifact_created", "stage_completed"]
    assert ref.id in events[1].artifact_ids


def test_bracket_wires_derived_from_edges(tmp_path: Path) -> None:
    """The returned ref's parent_ids become derived_from lineage edges."""
    ctx = _make_ctx(tmp_path)
    parent = asyncio.run(run_stage_bracketed(ctx, SeedStage()))
    child = asyncio.run(run_stage_bracketed(ctx, ChildStage(parent.id)))

    ancestors = [r.kind for r in ctx.lineage_store.trace_backward(child.id)]
    assert ancestors == ["user_plan"]


# ─────────────────────────────────────────────────────────── failure paths


def test_bracket_persisted_failure_emits_artifact_then_failed(tmp_path: Path) -> None:
    """StagePersistedFailureError → artifact_created + edges then stage_failed."""
    ctx = _make_ctx(tmp_path)
    parent = ctx.artifact_store.put_json(
        kind="user_plan", obj={"step": "parent"}, created_by="seed", parent_ids=[]
    )
    with pytest.raises(StagePersistedFailureError):
        asyncio.run(run_stage_bracketed(ctx, _persist_then_raise_stage(parent.id)()))

    events = [e.type for e in ctx.event_log.list_events("run-test")]
    assert events == ["stage_started", "artifact_created", "stage_failed"]
    # The persisted failure report's lineage is recorded before the failure.
    descendants = ctx.lineage_store.trace_forward(parent.id)
    assert any(d.kind == "validation_report" for d in descendants)


def test_bracket_plain_failure_wraps_in_stage_execution_error(tmp_path: Path) -> None:
    """A plain exception surfaces as StageExecutionError with stage_failed."""
    ctx = _make_ctx(tmp_path)
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(run_stage_bracketed(ctx, PlainFailStage()))

    assert isinstance(exc.value.__cause__, RuntimeError)
    assert [e.type for e in ctx.event_log.list_events("run-test")] == [
        "stage_started",
        "stage_failed",
    ]


# ──────────────────────── pipeline context on lineage edges (stage + run_id)


def test_bracket_stamps_stage_and_run_id_on_lineage_edges(tmp_path: Path) -> None:
    """The audit bracket records WHICH stage of WHICH run wrote each edge."""
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


# ──────────────────────────────────────────── StageRunner thin-wrapper parity


def test_stage_runner_delegates_to_the_bracket(tmp_path: Path) -> None:
    """StageRunner.run_stage produces the identical audit to the bare bracket."""
    ctx = _make_ctx(tmp_path)
    ref = asyncio.run(StageRunner(ctx).run_stage(NoopStage()))

    events = ctx.event_log.list_events("run-test")
    assert [e.type for e in events] == ["stage_started", "artifact_created", "stage_completed"]
    assert ref.id in events[1].artifact_ids
