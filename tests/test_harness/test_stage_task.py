"""Audit-parity tests for ``StageTask`` (spec plan-mode-revival-01).

``StageTask`` lifts one harness ``Stage`` into a ``molexp.workflow``
Runnable (an object with ``async execute(ctx) -> str`` returning the
produced ``ArtifactRef.id``). The adapter MUST emit the exact same audit
as the legacy ``StageRunner``:

- success path → ``stage_started`` / ``artifact_created`` / ``stage_completed``
- ``StagePersistedFailureError`` → ``stage_started`` / ``artifact_created``
  / ``stage_failed`` plus the persisted ref's ``derived_from`` edges
- plain exception → ``stage_started`` / ``stage_failed`` wrapped in
  ``StageExecutionError``

Parity is asserted by running identical stages through ``StageRunner`` and
through ``StageTask`` against *fresh* stores, then comparing the EventLog
event-type sequence and the ProvenanceStore ``derived_from`` edge set.

These map to acceptance ac-001..ac-005. They are RED until
``src/molexp/harness/core/stage_task.py`` exists and exports ``StageTask``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import ArtifactRef
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

# ───────────────────────────────────────────────────────── fixtures / helpers


def _make_ctx(root: Path, *, run_id: str = "run-test") -> HarnessRunContext:
    """Build a fresh HarnessRunContext backed by isolated on-disk stores."""
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteProvenanceStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id=run_id,
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        provenance_store=provenance,
    )


class _TaskContextStub:
    """Minimal stand-in for ``workflow.TaskContext`` exposing ``run_context``.

    ``StageTask.execute`` reads the live ``HarnessRunContext`` off
    ``ctx.run_context`` (the duck-typed payload threaded by
    ``Workflow.execute(run_context=...)``). The real TaskContext carries
    state/deps/inputs/config too; the adapter only consults ``run_context``,
    so a stub with just that attribute exercises the contract directly
    without spinning up the workflow engine.
    """

    def __init__(self, run_context: HarnessRunContext) -> None:
        self.run_context = run_context


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


# ─────────────────────────────────────────────────── ac-001: Runnable contract


def test_stage_task_is_a_workflow_runnable(tmp_path: Path) -> None:
    """ac-001: StageTask satisfies the workflow Runnable Protocol."""
    from molexp.harness.core.stage_task import StageTask
    from molexp.workflow.protocols import Runnable

    task = StageTask(NoopStage())
    assert isinstance(task, Runnable)


def test_stage_task_execute_returns_artifact_id(tmp_path: Path) -> None:
    """ac-001: execute() returns the produced ArtifactRef.id (a str)."""
    from molexp.harness.core.stage_task import StageTask

    ctx = _make_ctx(tmp_path)
    task = StageTask(NoopStage())

    produced = asyncio.run(task.execute(_TaskContextStub(ctx)))

    assert isinstance(produced, str)
    # The returned id must reference an artifact that actually landed in the store.
    ref = ctx.artifact_store.get_ref(produced)
    assert ref.id == produced
    assert ref.kind == "log"


# ──────────────────────────────────── ac-002 / ac-003: success-path parity


def test_stage_task_event_sequence_matches_stage_runner(tmp_path: Path) -> None:
    """ac-002: identical EventLog event-type sequence to StageRunner."""
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.core.stage_task import StageTask

    # Legacy path against its own fresh store.
    runner_ctx = _make_ctx(tmp_path / "runner", run_id="run-runner")
    asyncio.run(StageRunner(runner_ctx).run_stage(SeedStage()))
    runner_events = [e.type for e in runner_ctx.event_log.list_events("run-runner")]

    # StageTask path against a separate fresh store.
    task_ctx = _make_ctx(tmp_path / "task", run_id="run-task")
    asyncio.run(StageTask(SeedStage()).execute(_TaskContextStub(task_ctx)))
    task_events = [e.type for e in task_ctx.event_log.list_events("run-task")]

    assert task_events == ["stage_started", "artifact_created", "stage_completed"]
    assert task_events == runner_events


def test_stage_task_derived_from_edges_match_stage_runner(tmp_path: Path) -> None:
    """ac-003: identical derived_from provenance edge set to StageRunner."""
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.core.stage_task import StageTask

    # Two-stage chain through StageRunner.
    runner_ctx = _make_ctx(tmp_path / "runner", run_id="run-runner")
    parent_r = asyncio.run(StageRunner(runner_ctx).run_stage(SeedStage()))
    child_r = asyncio.run(StageRunner(runner_ctx).run_stage(ChildStage(parent_r.id)))
    runner_ancestors = [r.kind for r in runner_ctx.provenance_store.trace_backward(child_r.id)]

    # Same chain through StageTask against a fresh store.
    task_ctx = _make_ctx(tmp_path / "task", run_id="run-task")
    parent_id = asyncio.run(StageTask(SeedStage()).execute(_TaskContextStub(task_ctx)))
    child_id = asyncio.run(StageTask(ChildStage(parent_id)).execute(_TaskContextStub(task_ctx)))
    task_ancestors = [r.kind for r in task_ctx.provenance_store.trace_backward(child_id)]

    assert task_ancestors == ["user_plan"]
    assert task_ancestors == runner_ancestors


# ─────────────────────────── ac-004: persisted-failure + plain-failure parity


def test_stage_task_persisted_failure_emits_artifact_then_failed(tmp_path: Path) -> None:
    """ac-004: StagePersistedFailureError → artifact_created + edges then stage_failed."""
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.core.stage_task import StageTask

    # Baseline through StageRunner.
    runner_ctx = _make_ctx(tmp_path / "runner", run_id="run-runner")
    parent_r = runner_ctx.artifact_store.put_json(
        kind="user_plan", obj={"step": "parent"}, created_by="seed", parent_ids=[]
    )
    with pytest.raises(StageExecutionError):
        asyncio.run(StageRunner(runner_ctx).run_stage(_persist_then_raise_stage(parent_r.id)()))
    runner_events = [e.type for e in runner_ctx.event_log.list_events("run-runner")]

    # StageTask path against a fresh store.
    task_ctx = _make_ctx(tmp_path / "task", run_id="run-task")
    parent_t = task_ctx.artifact_store.put_json(
        kind="user_plan", obj={"step": "parent"}, created_by="seed", parent_ids=[]
    )
    with pytest.raises(StageExecutionError):
        asyncio.run(
            StageTask(_persist_then_raise_stage(parent_t.id)()).execute(_TaskContextStub(task_ctx))
        )
    task_events = [e.type for e in task_ctx.event_log.list_events("run-task")]

    assert task_events == ["stage_started", "artifact_created", "stage_failed"]
    assert task_events == runner_events
    # The persisted failure report's lineage is recorded before the failure.
    descendants = task_ctx.provenance_store.trace_forward(parent_t.id)
    assert any(d.kind == "validation_report" for d in descendants)


def test_stage_task_plain_failure_wraps_in_stage_execution_error(tmp_path: Path) -> None:
    """ac-004: a plain exception surfaces as StageExecutionError with stage_failed."""
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.core.stage_task import StageTask

    runner_ctx = _make_ctx(tmp_path / "runner", run_id="run-runner")
    with pytest.raises(StageExecutionError) as runner_exc:
        asyncio.run(StageRunner(runner_ctx).run_stage(PlainFailStage()))
    assert isinstance(runner_exc.value.__cause__, RuntimeError)
    runner_events = [e.type for e in runner_ctx.event_log.list_events("run-runner")]

    task_ctx = _make_ctx(tmp_path / "task", run_id="run-task")
    with pytest.raises(StageExecutionError) as task_exc:
        asyncio.run(StageTask(PlainFailStage()).execute(_TaskContextStub(task_ctx)))
    assert isinstance(task_exc.value.__cause__, RuntimeError)
    task_events = [e.type for e in task_ctx.event_log.list_events("run-task")]

    assert task_events == ["stage_started", "stage_failed"]
    assert task_events == runner_events


# ──────────────────────────────────── ac-005: StageRunner delegation unchanged


def test_stage_runner_still_passes_after_helper_lift(tmp_path: Path) -> None:
    """ac-005: StageRunner (now delegating to the shared bracket helper) is unchanged.

    The lifted shared audit-bracket helper must keep StageRunner's behaviour
    byte-identical: NoopStage still emits exactly the three success events.
    """
    from molexp.harness.core.stage_runner import StageRunner

    ctx = _make_ctx(tmp_path)
    ref = asyncio.run(StageRunner(ctx).run_stage(NoopStage()))

    events = ctx.event_log.list_events("run-test")
    assert [e.type for e in events] == ["stage_started", "artifact_created", "stage_completed"]
    assert ref.id in events[1].artifact_ids
