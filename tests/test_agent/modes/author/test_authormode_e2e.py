"""End-to-end ``AuthorMode.run`` driven by a ``ScriptedRouter`` (ac-009)."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.harness.events import (
    ArtifactWrittenEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
)
from molexp.agent.modes._planning import PlanState
from molexp.agent.modes.author._mode import AuthorMode, AuthorModeConfig
from molexp.agent.modes.author.codegen import GeneratedModule, TaskImplDraft, TaskIRBrief
from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff

from .conftest import ScriptedRouter, make_harness


def _module(node_id: str) -> GeneratedModule:
    """Test-codegen factory — tests still use the freeform ``GeneratedModule`` schema."""
    task_id = node_id.rsplit("/", 1)[-1]
    return GeneratedModule(
        task_id=task_id,
        source=f"def test_{task_id}() -> None:\n    assert True\n",
    )


def _impl_draft(node_id: str) -> TaskImplDraft:
    """Impl-codegen factory — body-only draft; the assembler wraps it."""
    del node_id
    return TaskImplDraft(imports=(), body="pass")


def _scripted_router() -> ScriptedRouter:
    router = ScriptedRouter(
        [
            TaskIRBrief(task_id="prepare", responsibility="prepare"),
            TaskIRBrief(task_id="run", responsibility="run"),
        ]
    )
    router.register_factory(GeneratedModule, _module)
    router.register_factory(TaskImplDraft, _impl_draft)
    return router


async def _drain(mode: AuthorMode, harness: object) -> list[object]:
    events: list[object] = []
    async for event in mode.run(harness=harness, user_input="materialize"):  # type: ignore[arg-type]
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_authormode_e2e_yields_ready_for_run_handoff(
    tmp_path: Path,
    plan_folder: object,
    approved_handoff: object,
) -> None:
    harness, _ = make_harness(_scripted_router(), scratch_dir=tmp_path / "scratch")
    mode = AuthorMode(
        config=AuthorModeConfig(debug_attempts=1),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=approved_handoff,  # type: ignore[arg-type]
    )
    events = await _drain(mode, harness)

    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "ready_for_run"

    handoff_dump = terminal.result["mode_state"]["handoff"]
    handoff = MaterializedWorkspaceHandoff.model_validate(handoff_dump)
    assert handoff.plan_graph.state is PlanState.ready_for_run
    assert handoff.entrypoint_module == "experiment.workflow"
    assert handoff.entrypoint_symbol == "create_workflow"


@pytest.mark.asyncio
async def test_authormode_e2e_materializes_workspace_files(
    tmp_path: Path,
    plan_folder: object,
    approved_handoff: object,
) -> None:
    harness, _ = make_harness(_scripted_router(), scratch_dir=tmp_path / "scratch")
    mode = AuthorMode(
        config=AuthorModeConfig(debug_attempts=1),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=approved_handoff,  # type: ignore[arg-type]
    )
    await _drain(mode, harness)

    root = Path(str(plan_folder.path()))  # type: ignore[attr-defined]
    assert (root / "ir" / "workflow.yaml").exists()
    assert (root / "src" / "experiment" / "workflow.py").exists()
    assert (root / "src" / "experiment" / "tasks" / "prepare.py").exists()
    assert (root / "tests" / "test_prepare.py").exists()
    assert (root / "manifest.yaml").exists()


@pytest.mark.asyncio
async def test_authormode_e2e_emits_lifecycle_events(
    tmp_path: Path,
    plan_folder: object,
    approved_handoff: object,
) -> None:
    harness, sink_events = make_harness(_scripted_router(), scratch_dir=tmp_path / "scratch")
    mode = AuthorMode(
        config=AuthorModeConfig(debug_attempts=1),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=approved_handoff,  # type: ignore[arg-type]
    )
    yielded = await _drain(mode, harness)

    # The harness emits lifecycle events to its sink; the mode yields the
    # terminal ModeCompletedEvent. AgentRunner interleaves both — here we
    # check the union.
    all_events = list(sink_events) + yielded
    kinds = {getattr(e, "kind", "") for e in all_events}
    assert "mode_started" in kinds
    assert "mode_completed" in kinds
    assert "approval_requested" in kinds
    assert "approval_decided" in kinds
    assert "stage_started" in kinds
    assert "artifact_written" in kinds
    assert any(isinstance(e, ModeStartedEvent) for e in sink_events)
    assert any(isinstance(e, ArtifactWrittenEvent) for e in all_events)


@pytest.mark.asyncio
async def test_authormode_e2e_transitions_plan_to_ready(
    tmp_path: Path,
    plan_folder: object,
    approved_handoff: object,
) -> None:
    harness, _ = make_harness(_scripted_router(), scratch_dir=tmp_path / "scratch")
    mode = AuthorMode(
        config=AuthorModeConfig(debug_attempts=1),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=approved_handoff,  # type: ignore[arg-type]
    )
    await _drain(mode, harness)
    assert plan_folder.plan_state is PlanState.ready_for_run  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_authormode_without_handoff_fails(
    tmp_path: Path,
    plan_folder: object,
) -> None:
    harness, _ = make_harness(ScriptedRouter(), scratch_dir=tmp_path / "scratch")
    mode = AuthorMode(
        config=AuthorModeConfig(),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=None,
    )
    events = await _drain(mode, harness)
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "failed"
