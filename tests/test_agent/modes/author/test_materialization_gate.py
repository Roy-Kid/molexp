"""Tests for the ``approve_materialization`` gate + lifecycle (ac-007)."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.harness.events import ModeCompletedEvent
from molexp.agent.harness.hooks import HookContext, HookPoint
from molexp.agent.modes._planning import (
    IllegalPlanTransitionError,
    PlanState,
    assert_legal_transition,
)
from molexp.agent.modes.author._mode import AuthorMode, AuthorModeConfig
from molexp.agent.modes.author.codegen import GeneratedModule, TaskImplDraft, TaskIRBrief
from molexp.agent.review import ReviewDecision

from .conftest import ScriptedRouter, make_harness


def _reject_hook(ctx: HookContext):
    async def _hook(_ctx: HookContext) -> ReviewDecision:
        return ReviewDecision(approved=False, reason="reviewer rejected")

    return _hook


async def _drain(mode: AuthorMode, harness: object) -> list[object]:
    events: list[object] = []
    async for event in mode.run(harness=harness, user_input="materialize"):  # type: ignore[arg-type]
        events.append(event)
    return events


# ── assert_legal_transition ──────────────────────────────────────────────


def test_assert_legal_transition_accepts_pipeline_path() -> None:
    assert_legal_transition(PlanState.approved, PlanState.materializing)
    assert_legal_transition(PlanState.materializing, PlanState.validating)
    assert_legal_transition(PlanState.validating, PlanState.ready_for_run)


def test_assert_legal_transition_rejects_illegal_jump() -> None:
    with pytest.raises(IllegalPlanTransitionError):
        assert_legal_transition(PlanState.approved, PlanState.ready_for_run)


# ── rejected gate ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejected_gate_writes_no_source_and_fails_plan(
    tmp_path: Path,
    plan_folder: object,
    approved_handoff: object,
) -> None:
    router = ScriptedRouter()
    harness, _ = make_harness(router, scratch_dir=tmp_path / "scratch")
    harness.hooks.register(HookPoint.before_approval, _reject_hook(None))

    mode = AuthorMode(
        config=AuthorModeConfig(),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=approved_handoff,  # type: ignore[arg-type]
    )
    events = await _drain(mode, harness)

    # No structured codegen call was ever made.
    assert router.calls == []
    # No src/ tree was written.
    assert not (Path(str(plan_folder.path())) / "src").exists()  # type: ignore[attr-defined]
    # The plan transitioned to failed.
    assert plan_folder.plan_state is PlanState.failed  # type: ignore[attr-defined]
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "failed"


# ── approved gate proceeds ───────────────────────────────────────────────


def _impl_draft(node_id: str) -> TaskImplDraft:
    """Impl path uses the constrained ``TaskImplDraft`` schema (body-only)."""
    del node_id
    return TaskImplDraft(imports=(), body="pass")


def _test_factory(node_id: str) -> GeneratedModule:
    task_id = node_id.rsplit("/", 1)[-1]
    return GeneratedModule(
        task_id=task_id,
        source=f"def test_{task_id}() -> None:\n    assert True\n",
    )


@pytest.mark.asyncio
async def test_approved_gate_proceeds_to_codegen(
    tmp_path: Path,
    plan_folder: object,
    approved_handoff: object,
) -> None:
    # Default (no rejecting hook) means the gate approves.
    router = ScriptedRouter(
        [
            TaskIRBrief(task_id="prepare", responsibility="prepare"),
            TaskIRBrief(task_id="run", responsibility="run"),
        ]
    )

    # Test generation goes through ``GeneratedModule``; impl generation
    # goes through the constrained ``TaskImplDraft`` schema. Register
    # both — without the TaskImplDraft factory the ScriptedRouter would
    # raise AssertionError, but AuthorMode.run silently swallows it,
    # leaving the gate-proceeds assertion to pass on a broken pipeline.
    router.register_factory(GeneratedModule, _test_factory)
    router.register_factory(TaskImplDraft, _impl_draft)
    harness, _ = make_harness(router, scratch_dir=tmp_path / "scratch")

    mode = AuthorMode(
        config=AuthorModeConfig(debug_attempts=1),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=approved_handoff,  # type: ignore[arg-type]
    )
    events = await _drain(mode, harness)

    # The gate let codegen run — the workflow IR was written.
    assert (Path(str(plan_folder.path())) / "ir" / "workflow.yaml").exists()  # type: ignore[attr-defined]
    # Impl generation actually completed — the per-task impl files
    # exist. A silently-broken codegen pipeline would have written
    # workflow.yaml in an earlier stage and then crashed before this.
    tasks_dir = Path(str(plan_folder.path())) / "src" / "experiment" / "tasks"  # type: ignore[attr-defined]
    assert (tasks_dir / "prepare.py").exists()
    assert (tasks_dir / "run.py").exists()
    # Terminal event reports plan reached ``ready_for_run`` (not failed).
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "ready_for_run"
