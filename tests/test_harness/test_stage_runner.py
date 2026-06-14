"""Tests for HarnessRunContext + Stage + StageRunner (Phase 1 core).

Locks the contract per spec §StageRunner:
- NoopStage → exactly [stage_started, artifact_created, stage_completed]
- FailingStage → exactly [stage_started, stage_failed] + raises StageExecutionError
- Two-stage pipeline materializes a derived_from edge from parent_ids
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------- fixtures


@pytest.fixture()
def ctx(tmp_path: Path):
    """Wire up a HarnessRunContext with the three Phase-1 stores."""
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db_path = tmp_path / "events.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-test",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=provenance,
    )


# ----------------------------------------------------------- success path


def test_noop_stage_emits_three_events_in_order(ctx) -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.core.stage_runner import StageRunner

    class NoopStage(Stage):
        name = "NoopStage"

        async def run(self, ctx):
            return ctx.artifact_store.put_json(
                kind="log",
                obj={"note": "hello"},
                created_by="NoopStage",
                parent_ids=[],
            )

    import asyncio

    runner = StageRunner(ctx)
    ref = asyncio.run(runner.run_stage(NoopStage()))

    events = ctx.event_log.list_events("run-test")
    assert [e.type for e in events] == [
        "stage_started",
        "artifact_created",
        "stage_completed",
    ]
    assert events[0].payload == {"stage": "NoopStage"}
    assert ref.id in events[1].artifact_ids
    assert events[2].payload == {"stage": "NoopStage"}


# ----------------------------------------------------------- failure path


def test_failing_stage_emits_failed_and_raises(ctx) -> None:
    import asyncio

    from molexp.harness.core.stage import Stage
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.errors import StageExecutionError

    class FailingStage(Stage):
        name = "FailingStage"

        async def run(self, ctx):
            raise RuntimeError("boom")

    runner = StageRunner(ctx)
    with pytest.raises(StageExecutionError) as exc_info:
        asyncio.run(runner.run_stage(FailingStage()))
    assert isinstance(exc_info.value.__cause__, RuntimeError)

    events = ctx.event_log.list_events("run-test")
    assert [e.type for e in events] == ["stage_started", "stage_failed"]
    assert "boom" in events[1].payload["error"]
    assert events[1].payload["stage"] == "FailingStage"


# --------------------------------------------- persist-then-raise contract


def test_persisted_failure_emits_artifact_created_and_edges_before_failing(ctx) -> None:
    """StagePersistedFailureError: runner emits artifact_created + edges then re-raises.

    Mirrors the always-persist-then-raise validator contract: a strict
    validator persists a parse-error ValidationReport, raises
    ``StagePersistedFailureError(persisted_ref, …)``, and the runner
    surfaces the persisted artifact + lineage in the audit trail before
    failing the stage.
    """
    import asyncio

    from molexp.harness.core.stage import Stage
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.errors import StageExecutionError, StagePersistedFailureError

    parent_ref = ctx.artifact_store.put_json(
        kind="user_plan",
        obj={"step": "parent"},
        created_by="seed",
        parent_ids=[],
    )

    class PersistThenRaiseStage(Stage):
        name = "PersistThenRaiseStage"

        async def run(self, ctx):
            failure_ref = ctx.artifact_store.put_json(
                kind="validation_report",
                obj={"passed": False, "violations": []},
                created_by="PersistThenRaiseStage",
                parent_ids=[parent_ref.id],
            )
            raise StagePersistedFailureError(failure_ref, "parse failed")

    runner = StageRunner(ctx)
    with pytest.raises(StageExecutionError):
        asyncio.run(runner.run_stage(PersistThenRaiseStage()))

    events = ctx.event_log.list_events("run-test")
    # Lineage event present BEFORE stage_failed; no stage_completed.
    types = [e.type for e in events]
    assert types == ["stage_started", "artifact_created", "stage_failed"]
    # And the provenance edge from parent → persisted report is recorded.
    descendants = ctx.lineage_store.trace_forward(parent_ref.id)
    assert any(d.kind == "validation_report" for d in descendants)


# --------------------------------------------------------- provenance wire


def test_two_stage_pipeline_materializes_derived_from_edge(ctx) -> None:
    import asyncio

    from molexp.harness.core.stage import Stage
    from molexp.harness.core.stage_runner import StageRunner

    class StageA(Stage):
        name = "StageA"

        async def run(self, ctx):
            return ctx.artifact_store.put_json(
                kind="user_plan",
                obj={"step": "A"},
                created_by="StageA",
                parent_ids=[],
            )

    class StageB(Stage):
        name = "StageB"

        def __init__(self, parent_id: str) -> None:
            self._parent_id = parent_id

        async def run(self, ctx):
            return ctx.artifact_store.put_json(
                kind="experiment_report",
                obj={"step": "B"},
                created_by="StageB",
                parent_ids=[self._parent_id],
            )

    runner = StageRunner(ctx)
    ref_a = asyncio.run(runner.run_stage(StageA()))
    ref_b = asyncio.run(runner.run_stage(StageB(parent_id=ref_a.id)))

    ancestors = ctx.lineage_store.trace_backward(ref_b.id)
    assert [r.id for r in ancestors] == [ref_a.id]


# --------------------------------------------------------------- contracts


def test_stage_is_an_abstract_class() -> None:
    """Stage MUST be abc.ABC — instantiating directly raises TypeError."""
    from molexp.harness.core.stage import Stage

    with pytest.raises(TypeError):
        Stage()  # type: ignore[abstract]


def test_harness_run_context_is_frozen(ctx) -> None:
    """HarnessRunContext is read-only at runtime."""
    with pytest.raises((AttributeError, TypeError)):
        ctx.run_id = "mutated"
