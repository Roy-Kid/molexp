"""Tests for the ``approve_execution`` gate (ac-003).

The ``approve_execution`` gate is RunMode's binding safety property: no
LLM-authored experiment code is imported or executed before a reviewer
clears it. A blocking :class:`ReviewPolicy`-shaped hook must keep the
plan at ``ready_for_run`` and prevent any ``Workflow.execute`` call; an
approving one must let the gate return ``ApprovalGate.approve_execution``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.events import (
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    ModeCompletedEvent,
)
from molexp.agent.modes._planning import ApprovalGate, PlanState
from molexp.agent.modes.run._mode import RunMode, RunModeConfig
from molexp.agent.modes.run.gate import approve_execution_gate, build_execution_view

from .conftest import (
    ScriptedRouter,
    StubWorkflow,
    approving_decision,
    blocking_decision,
    drain,
    make_handoff,
    make_harness,
    passing_result,
)

# ── gate consultation directly ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_approve_execution_gate_returns_decision_when_approved(
    tmp_path: Path,
) -> None:
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, sink = make_harness(ScriptedRouter(), approval=approving_decision())

    decision = await approve_execution_gate(handoff, harness=harness)

    assert decision.approved is True
    # The harness emitted the gate events under the approve_execution name.
    requested = [e for e in sink if isinstance(e, ApprovalRequestedEvent)]
    decided = [e for e in sink if isinstance(e, ApprovalDecidedEvent)]
    assert len(requested) == 1
    assert len(decided) == 1
    assert requested[0].gate == ApprovalGate.approve_execution.value
    assert decided[0].gate == ApprovalGate.approve_execution.value
    assert decided[0].approved is True


@pytest.mark.asyncio
async def test_approve_execution_gate_returns_block_when_rejected(
    tmp_path: Path,
) -> None:
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, _ = make_harness(ScriptedRouter(), approval=blocking_decision())

    decision = await approve_execution_gate(handoff, harness=harness)

    assert decision.approved is False


def test_build_execution_view_summarizes_handoff(tmp_path: Path) -> None:
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    view = build_execution_view(handoff)
    assert view.plan_id == "demo-plan"
    assert "create_workflow" in view.entrypoint
    assert "demo-plan" in view.summary


# ── blocking gate prevents all execution ─────────────────────────────────


@pytest.mark.asyncio
async def test_blocking_gate_imports_nothing_and_does_not_execute(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
) -> None:
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, _ = make_harness(ScriptedRouter(), approval=blocking_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )

    events = await drain(mode, harness)

    # The plan never left ready_for_run — RunMode did not enter `running`.
    assert plan_folder.plan_state is PlanState.ready_for_run  # type: ignore[attr-defined]
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "ready_for_run"
    # No RunReport — nothing was executed.
    assert "run" not in terminal.result["mode_state"]


@pytest.mark.asyncio
async def test_blocking_gate_runs_no_workflow_execute(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the gate is honoured, load_materialized_workflow is never reached.
    import molexp.agent.modes.run._mode as mode_module

    calls: list[object] = []

    def _spy_loader(handoff: object) -> object:
        calls.append(handoff)
        return StubWorkflow(result=passing_result(("prepare", "run")))

    monkeypatch.setattr(mode_module, "load_materialized_workflow", _spy_loader)

    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, _ = make_harness(ScriptedRouter(), approval=blocking_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    await drain(mode, harness)

    assert calls == []  # the workflow was never loaded


# ── require_execution_gate=False is the only bypass ──────────────────────


@pytest.mark.asyncio
async def test_gate_bypass_only_via_require_execution_gate_false(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import molexp.agent.modes.run._mode as mode_module

    stub = StubWorkflow(result=passing_result(("prepare", "run")))
    monkeypatch.setattr(mode_module, "load_materialized_workflow", lambda _h: stub)

    handoff = make_handoff(workspace_path=tmp_path / "ws")
    # No approval hook at all — and a blocking one would be ignored anyway.
    harness, sink = make_harness(ScriptedRouter())
    mode = RunMode(
        config=RunModeConfig(require_execution_gate=False),
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    events = await drain(mode, harness)

    # Bypass: no approval gate was opened at all.
    assert not [e for e in sink if isinstance(e, ApprovalRequestedEvent)]
    # Execution proceeded to completion.
    assert stub.execute_calls
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "completed"
