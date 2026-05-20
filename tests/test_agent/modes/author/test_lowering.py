"""Tests for ``MaterializedWorkspaceHandoff`` / ``AuthorModeConfig`` shape
and the ``PlanGraph`` → ``WorkflowContract`` lowering (ac-001, ac-004)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.agent.modes._planning import PlanState, PlanStepInput
from molexp.agent.modes.author._mode import AuthorModeConfig
from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff
from molexp.agent.modes.author.lowering import lower_plan_graph
from molexp.workflow import ValidationReport

from .conftest import make_capability_graph, make_plan_graph, make_step

# ── ac-001 — MaterializedWorkspaceHandoff shape ──────────────────────────


def test_handoff_has_exactly_the_prescribed_fields() -> None:
    expected = {
        "plan_id",
        "plan_graph",
        "experiment_workspace_path",
        "workflow_yaml_path",
        "entrypoint_module",
        "entrypoint_symbol",
        "source_root",
        "validation_report_snapshot",
        "materialization_approved_at",
    }
    assert set(MaterializedWorkspaceHandoff.model_fields) == expected


def test_handoff_is_frozen() -> None:
    handoff = MaterializedWorkspaceHandoff(
        plan_id="p",
        plan_graph=make_plan_graph(state=PlanState.ready_for_run),
        experiment_workspace_path=Path("/tmp/p"),
        workflow_yaml_path=Path("/tmp/p/ir/workflow.yaml"),
        entrypoint_module="experiment.workflow",
        entrypoint_symbol="create_workflow",
        source_root=Path("/tmp/p/src"),
        validation_report_snapshot=ValidationReport(ok=True),
        materialization_approved_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="frozen"):
        handoff.plan_id = "other"  # type: ignore[misc]


def test_handoff_validation_report_is_workflow_type() -> None:
    handoff = MaterializedWorkspaceHandoff(
        plan_id="p",
        plan_graph=make_plan_graph(state=PlanState.ready_for_run),
        experiment_workspace_path=Path("/tmp/p"),
        workflow_yaml_path=Path("/tmp/p/ir/workflow.yaml"),
        entrypoint_module="experiment.workflow",
        entrypoint_symbol="create_workflow",
        source_root=Path("/tmp/p/src"),
        validation_report_snapshot=ValidationReport(ok=True),
        materialization_approved_at=datetime.now(UTC),
    )
    assert isinstance(handoff.validation_report_snapshot, ValidationReport)


# ── ac-001 — AuthorModeConfig shape ──────────────────────────────────────


def test_author_mode_config_defaults() -> None:
    config = AuthorModeConfig()
    assert config.debug_attempts >= 1
    assert config.subprocess_timeout_seconds > 0


def test_author_mode_config_is_frozen() -> None:
    config = AuthorModeConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.debug_attempts = 9  # type: ignore[misc]


# ── ac-004 — lowering ────────────────────────────────────────────────────


def test_lowering_produces_valid_contract() -> None:
    result = lower_plan_graph(make_plan_graph())
    assert result.validation_report.ok
    assert result.ok


def test_lowering_sets_compiled_contract_ref() -> None:
    plan = make_plan_graph()
    assert plan.compiled_contract_ref is None
    result = lower_plan_graph(plan)
    assert result.plan_graph.compiled_contract_ref == result.contract.workflow_id


def test_lowering_maps_every_step_to_a_task_io() -> None:
    plan = make_plan_graph()
    result = lower_plan_graph(plan)
    task_ids = {tio.task_id for tio in result.contract.task_io}
    assert task_ids == {step.id for step in plan.steps}


def test_lowering_carries_input_sources() -> None:
    result = lower_plan_graph(make_plan_graph())
    run_io = next(tio for tio in result.contract.task_io if tio.task_id == "run")
    assert run_io.inputs[0].name == "payload"
    assert run_io.inputs[0].source == "prepare"


def test_lowering_surfaces_residual_issues_for_broken_plan() -> None:
    # A 'run' step whose input sources from a non-existent step.
    prepare = make_step("prepare", outputs=("payload",))
    broken = make_step(
        "run",
        depends_on=("missing",),
        inputs=(PlanStepInput(name="payload", source_step="missing"),),
    )
    plan = make_plan_graph().model_copy(update={"steps": (prepare, broken)})
    result = lower_plan_graph(plan)
    # The dangling source is surfaced, not silently dropped.
    assert not result.ok
    assert result.normalize_report.issues or not result.validation_report.ok


def test_lowering_does_not_mutate_input_plan() -> None:
    plan = make_plan_graph()
    lower_plan_graph(plan)
    assert plan.compiled_contract_ref is None


def test_capability_graph_round_trips_through_lowering() -> None:
    # The capability graph is not lowered, but the lowering must not need it.
    _ = make_capability_graph()
    result = lower_plan_graph(make_plan_graph())
    assert result.contract.workflow_id.startswith("wf_")
