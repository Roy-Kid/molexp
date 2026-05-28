"""End-to-end ``RunMode.run`` pipeline tests (ac-001 / ac-004 / ac-011).

Drives :class:`RunMode` against a stubbed :class:`Workflow` and an
approving / blocking review hook — no live HPC, no live LLM. Covers the
happy path (``ready_for_run → running → completed`` with a well-formed
:class:`RunReport`), the unrecoverable-failure path, the
re-materialization escalation (``running → needs_clarification``), and
:class:`RunFolder` persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.events import (
    ModeCompletedEvent,
    ModeStartedEvent,
    RepairProposedEvent,
    StageStartedEvent,
)
from molexp.agent.mode import AgentMode
from molexp.agent.modes import RunMode, RunModeConfig
from molexp.agent.modes._planning import PlanState
from molexp.agent.modes.run.run_folder import AGENT_RUN_KIND, RunFolder, RunReport

from .conftest import (
    ScriptedRouter,
    StubWorkflow,
    approving_decision,
    drain,
    failing_result,
    make_handoff,
    make_harness,
    passing_result,
)


def _patch_loader(monkeypatch: pytest.MonkeyPatch, workflow: object) -> None:
    """Make ``load_materialized_workflow`` return ``workflow``."""
    import molexp.agent.modes.run._mode as mode_module

    monkeypatch.setattr(mode_module, "load_materialized_workflow", lambda _h: workflow)


# ── ac-001 — RunMode is an AgentMode ─────────────────────────────────────


def test_run_mode_is_agent_mode_subclass() -> None:
    assert issubclass(RunMode, AgentMode)
    assert RunMode.name == "run"


def test_run_mode_exported_from_modes() -> None:
    import molexp.agent.modes as modes

    assert "RunMode" in modes.__all__
    assert "RunModeConfig" in modes.__all__


# ── ac-004 — happy path ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_mode_passing_workflow_completes(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWorkflow(result=passing_result(("prepare", "run")))
    _patch_loader(monkeypatch, stub)

    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, sink = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    events = await drain(mode, harness)

    # The plan transitioned ready_for_run -> running -> completed.
    assert plan_folder.plan_state is PlanState.completed  # type: ignore[attr-defined]

    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "completed"

    # mode_state["run"] holds a well-formed RunReport.
    report = RunReport.model_validate(terminal.result["mode_state"]["run"])
    assert report.plan_id == "demo-plan"
    assert report.status == "completed"
    assert report.progress.all_succeeded is True

    # Lifecycle events were emitted.
    kinds = {getattr(e, "kind", "") for e in list(sink) + events}
    assert {"mode_started", "mode_completed", "stage_started", "stage_completed"} <= kinds
    assert "approval_requested" in kinds
    assert any(isinstance(e, ModeStartedEvent) for e in sink)


@pytest.mark.asyncio
async def test_run_mode_emits_workflow_stages(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWorkflow(result=passing_result(("prepare", "run")))
    _patch_loader(monkeypatch, stub)
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, sink = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    await drain(mode, harness)

    stage_names = {e.stage_name for e in sink if isinstance(e, StageStartedEvent)}
    assert "LoadMaterializedWorkflow" in stage_names
    assert "ExecuteWorkflow" in stage_names


# ── ac-011 — RunFolder persistence ───────────────────────────────────────


@pytest.mark.asyncio
async def test_run_folder_persists_run_report(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWorkflow(result=passing_result(("prepare", "run")))
    _patch_loader(monkeypatch, stub)
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, _ = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    await drain(mode, harness)

    # A RunFolder (kind "agent.run") was mounted under the plan.
    run_folder = plan_folder.get_folder(  # type: ignore[attr-defined]
        "run-demo-plan", cls=RunFolder
    )
    assert run_folder.kind == AGENT_RUN_KIND
    assert run_folder.plan_id == "demo-plan"

    # run_report.yaml exists and round-trips through RunReport.model_validate.
    report_path = Path(str(run_folder.path())) / "run_report.yaml"
    assert report_path.exists()
    reloaded = run_folder.read_run_report()
    assert isinstance(reloaded, RunReport)
    assert reloaded.plan_id == "demo-plan"
    assert reloaded.status == "completed"


# ── unrecoverable failure → PlanDiff ─────────────────────────────────────


@pytest.mark.asyncio
async def test_run_mode_failure_emits_repair_and_escalates(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `run` fails (only `prepare` produced an output).
    stub = StubWorkflow(result=failing_result(("prepare",)))
    _patch_loader(monkeypatch, stub)
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, sink = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    events = await drain(mode, harness)

    # A repair_proposed event was emitted.
    repair_events = [e for e in list(sink) + events if isinstance(e, RepairProposedEvent)]
    assert len(repair_events) == 1
    assert repair_events[0].failed_invariant

    # The diff needs re-materialization -> running -> needs_clarification.
    assert plan_folder.plan_state is PlanState.needs_clarification  # type: ignore[attr-defined]

    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "needs_clarification"

    report = RunReport.model_validate(terminal.result["mode_state"]["run"])
    assert report.status == "needs_clarification"
    assert len(report.repair_diffs) == 1
    assert report.escalation is not None
    assert report.escalation.requires_rematerialization is True
    assert report.escalation.target_mode == "author"


@pytest.mark.asyncio
async def test_run_mode_failure_persists_escalation(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWorkflow(result=failing_result(("prepare",)))
    _patch_loader(monkeypatch, stub)
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, _ = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    await drain(mode, harness)

    run_folder = plan_folder.get_folder(  # type: ignore[attr-defined]
        "run-demo-plan", cls=RunFolder
    )
    escalation_path = Path(str(run_folder.path())) / "repairs" / "escalation.json"
    assert escalation_path.exists()


@pytest.mark.asyncio
async def test_run_mode_workflow_raises_ends_failed(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWorkflow(raises=RuntimeError("execution blew up"))
    _patch_loader(monkeypatch, stub)
    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, _ = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    events = await drain(mode, harness)

    # A raised workflow that produces a structural diff still escalates.
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] in {
        "failed",
        "needs_clarification",
    }


# ── load failure ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_mode_load_failure_ends_failed(
    tmp_path: Path,
    plan_folder: object,
    experiment: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import molexp.agent.modes.run._mode as mode_module
    from molexp.agent.modes.run.loader import WorkflowLoadError

    def _bad_loader(_h: object) -> object:
        raise WorkflowLoadError("entrypoint missing")

    monkeypatch.setattr(mode_module, "load_materialized_workflow", _bad_loader)

    handoff = make_handoff(workspace_path=tmp_path / "ws")
    harness, sink = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=handoff,
    )
    events = await drain(mode, harness)

    assert plan_folder.plan_state is PlanState.failed  # type: ignore[attr-defined]
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "failed"
    kinds = {getattr(e, "kind", "") for e in sink}
    assert "error" in kinds


# ── no handoff ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_mode_without_handoff_stays_ready(
    plan_folder: object,
    experiment: object,
) -> None:
    harness, _ = make_harness(ScriptedRouter(), approval=approving_decision())
    mode = RunMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        experiment=experiment,  # type: ignore[arg-type]
        handoff=None,
    )
    events = await drain(mode, harness)

    assert plan_folder.plan_state is PlanState.ready_for_run  # type: ignore[attr-defined]
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    assert terminal.result["mode_state"]["plan_state"] == "ready_for_run"


# ── config ───────────────────────────────────────────────────────────────


def test_run_mode_config_defaults() -> None:
    config = RunModeConfig()
    assert config.require_execution_gate is True
    assert config.max_repair_escalations == 1
    assert config.retry_backoff_seconds == 0.0
