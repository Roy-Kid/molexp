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
from molexp.agent.modes.author.codegen import GeneratedModule, TaskIRBrief
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


def _impl_factory(node_id: str) -> GeneratedModule:
    task_id = node_id.rsplit("/", 1)[-1]
    return GeneratedModule(
        task_id=task_id,
        source=(
            "from molexp.workflow import Task\n\n\n"
            f"class {task_id.title()}(Task):\n"
            "    async def execute(self, ctx):\n"
            "        return None\n"
        ),
    )


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

    def _module(node_id: str) -> GeneratedModule:
        if node_id.startswith("GenerateTaskTests"):
            return _test_factory(node_id)
        return _impl_factory(node_id)

    router.register_factory(GeneratedModule, _module)
    harness, _ = make_harness(router, scratch_dir=tmp_path / "scratch")

    mode = AuthorMode(
        config=AuthorModeConfig(debug_attempts=1),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        handoff=approved_handoff,  # type: ignore[arg-type]
    )
    await _drain(mode, harness)

    # The gate let codegen run — the workflow IR was written.
    assert (Path(str(plan_folder.path())) / "ir" / "workflow.yaml").exists()  # type: ignore[attr-defined]
