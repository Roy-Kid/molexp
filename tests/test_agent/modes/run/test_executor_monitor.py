"""Tests for ``load_materialized_workflow`` + ``RunExecutor`` + ``StepMonitor``.

Covers ac-005 (loader imports the entrypoint and type-asserts it),
ac-006 (``StepMonitor`` projects execution onto the typed plan steps),
and the executor binding a :class:`Workflow` to a workspace ``Run``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from molexp.agent.modes.run.executor import RunExecutor
from molexp.agent.modes.run.loader import WorkflowLoadError, load_materialized_workflow
from molexp.agent.modes.run.monitor import RunProgress, StepMonitor, StepStatus

from .conftest import (
    StubWorkflow,
    failing_result,
    make_handoff,
    make_plan_graph,
    passing_result,
)

# ── load_materialized_workflow ───────────────────────────────────────────

_VALID_ENTRYPOINT = """
from molexp.workflow import WorkflowBuilder


def create_workflow():
    return WorkflowBuilder(name="materialized").build()
"""

_NOT_A_WORKFLOW = """
def create_workflow():
    return {"not": "a workflow"}
"""

_DIRECT_WORKFLOW = """
from molexp.workflow import WorkflowBuilder

WORKFLOW = WorkflowBuilder(name="direct").build()
"""


def _write_module(root: Path, package: str, module: str, body: str) -> None:
    """Write ``package/module.py`` (with an ``__init__.py``) under ``root``."""
    pkg_dir = root / package
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / f"{module}.py").write_text(body)


@pytest.fixture(autouse=True)
def _clean_modules() -> None:
    """Drop fixture modules from ``sys.modules`` between tests."""
    yield
    for name in list(sys.modules):
        if name.startswith("experiment"):
            del sys.modules[name]


def test_load_materialized_workflow_returns_workflow(tmp_path: Path) -> None:
    from molexp.workflow import Workflow

    source_root = tmp_path / "src"
    _write_module(source_root, "experiment", "workflow", _VALID_ENTRYPOINT)
    handoff = make_handoff(workspace_path=tmp_path / "ws", source_root=source_root)

    workflow = load_materialized_workflow(handoff)

    assert isinstance(workflow, Workflow)
    assert workflow.name == "materialized"


def test_load_materialized_workflow_accepts_direct_workflow_symbol(
    tmp_path: Path,
) -> None:
    from molexp.workflow import Workflow

    source_root = tmp_path / "src"
    _write_module(source_root, "experiment", "workflow", _DIRECT_WORKFLOW)
    handoff = make_handoff(
        workspace_path=tmp_path / "ws",
        source_root=source_root,
        entrypoint_symbol="WORKFLOW",
    )

    workflow = load_materialized_workflow(handoff)
    assert isinstance(workflow, Workflow)


def test_load_materialized_workflow_raises_on_missing_module(tmp_path: Path) -> None:
    handoff = make_handoff(
        workspace_path=tmp_path / "ws",
        source_root=tmp_path / "src",
        entrypoint_module="experiment.nonexistent_module",
    )
    with pytest.raises(WorkflowLoadError, match="cannot import entrypoint module"):
        load_materialized_workflow(handoff)


def test_load_materialized_workflow_raises_on_missing_symbol(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    _write_module(source_root, "experiment", "workflow", _VALID_ENTRYPOINT)
    handoff = make_handoff(
        workspace_path=tmp_path / "ws",
        source_root=source_root,
        entrypoint_symbol="not_there",
    )
    with pytest.raises(WorkflowLoadError, match="no symbol"):
        load_materialized_workflow(handoff)


def test_load_materialized_workflow_raises_when_not_a_workflow(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    _write_module(source_root, "experiment", "workflow", _NOT_A_WORKFLOW)
    handoff = make_handoff(workspace_path=tmp_path / "ws", source_root=source_root)
    with pytest.raises(WorkflowLoadError, match=r"not a molexp\.workflow\.Workflow"):
        load_materialized_workflow(handoff)


# ── StepMonitor / RunProgress projection ─────────────────────────────────


def test_step_monitor_one_row_per_plan_step() -> None:
    plan = make_plan_graph()
    monitor = StepMonitor(plan)
    progress = monitor.snapshot()

    assert isinstance(progress, RunProgress)
    assert len(progress.steps) == len(plan.steps)
    assert tuple(s.step_id for s in progress.steps) == tuple(s.id for s in plan.steps)
    # Initial rows are pending.
    assert all(s.status is StepStatus.pending for s in progress.steps)


def test_step_monitor_marks_success_and_failure() -> None:
    plan = make_plan_graph()
    monitor = StepMonitor(plan)
    monitor.mark_running("prepare")
    monitor.mark_succeeded("prepare")
    monitor.mark_failed("run", error_ref="RuntimeError: boom")

    progress = monitor.snapshot()
    prepare = progress.step("prepare")
    run = progress.step("run")
    assert prepare is not None and prepare.status is StepStatus.succeeded
    assert prepare.started_at is not None and prepare.finished_at is not None
    assert run is not None and run.status is StepStatus.failed
    assert run.error_ref == "RuntimeError: boom"
    assert progress.failed_step_ids == ("run",)
    assert progress.all_succeeded is False


def test_step_monitor_all_succeeded() -> None:
    plan = make_plan_graph()
    monitor = StepMonitor(plan)
    for step in plan.steps:
        monitor.mark_succeeded(step.id)
    assert monitor.snapshot().all_succeeded is True


def test_step_monitor_rejects_unknown_step() -> None:
    monitor = StepMonitor(make_plan_graph())
    with pytest.raises(KeyError):
        monitor.mark_running("nonexistent")


# ── RunExecutor ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_executor_binds_workflow_to_run_and_projects_progress(
    experiment: object,
) -> None:
    plan = make_plan_graph()
    run = experiment.add_run(id="exec-run")  # type: ignore[attr-defined]
    stub = StubWorkflow(result=passing_result(("prepare", "run")))
    executor = RunExecutor(workflow=stub, run=run, plan_graph=plan)  # type: ignore[arg-type]

    outcome = await executor.execute()

    # The workflow was driven through `execute(run_context=...)`.
    assert len(stub.execute_calls) == 1
    assert "run_context" in stub.execute_calls[0]
    # Progress projection — one StepProgress per PlanStep, all succeeded.
    assert outcome.succeeded is True
    assert outcome.status == "completed"
    assert len(outcome.progress.steps) == len(plan.steps)
    assert outcome.progress.all_succeeded is True


@pytest.mark.asyncio
async def test_run_executor_projects_partial_failure(experiment: object) -> None:
    plan = make_plan_graph()
    run = experiment.add_run(id="exec-run-fail")  # type: ignore[attr-defined]
    # Only `prepare` produced an output; `run` failed.
    stub = StubWorkflow(result=failing_result(("prepare",)))
    executor = RunExecutor(workflow=stub, run=run, plan_graph=plan)  # type: ignore[arg-type]

    outcome = await executor.execute()

    assert outcome.succeeded is False
    assert outcome.status == "failed"
    assert outcome.progress.step("prepare").status is StepStatus.succeeded  # type: ignore[union-attr]
    assert outcome.progress.step("run").status is StepStatus.failed  # type: ignore[union-attr]
    assert outcome.progress.failed_step_ids == ("run",)


@pytest.mark.asyncio
async def test_run_executor_records_raised_workflow_as_failed(
    experiment: object,
) -> None:
    plan = make_plan_graph()
    run = experiment.add_run(id="exec-run-raise")  # type: ignore[attr-defined]
    stub = StubWorkflow(raises=RuntimeError("workflow blew up"))
    executor = RunExecutor(workflow=stub, run=run, plan_graph=plan)  # type: ignore[arg-type]

    outcome = await executor.execute()

    assert outcome.succeeded is False
    assert outcome.status == "failed"
    assert outcome.error_type == "RuntimeError"
    assert "workflow blew up" in outcome.error_message
