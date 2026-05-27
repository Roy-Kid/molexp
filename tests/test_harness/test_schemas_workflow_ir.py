"""Tests for WorkflowIR / TaskIR / DependencyEdge / ExpectedOutput (Phase 3 §4.6).

Locks the wire format:
- frozen pydantic round-trip
- ExpectedOutput.kind is ArtifactKind (Phase-1 Literal)
- TaskIR.inputs / constraints accept dict[str, ParameterValue]
- Defaults are independent (no shared mutable state)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _make_task(task_id: str = "t1") -> TaskIR:  # noqa: F821 — Phase-3 type loaded inside body
    from molexp.harness.schemas.workflow_ir import TaskIR

    return TaskIR(
        id=task_id,
        name="Build polymer",
        purpose="Generate atomistic configuration",
        task_type="molecule_builder",
        inputs={},
        outputs={"structure": "structure.pdb"},
    )


def test_workflow_ir_round_trip() -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import (
        DependencyEdge,
        ExpectedOutput,
        TaskIR,
        WorkflowIR,
    )

    ir = WorkflowIR(
        id="wf-001",
        name="water_nemd",
        objective="Compute ionic mobility under field",
        inputs={
            "temperature_K": ParameterValue(value=300.0, source="user_provided"),
        },
        tasks=[
            TaskIR(
                id="build",
                name="Build system",
                purpose="Pack water box",
                task_type="molecule_builder",
                inputs={"temperature_K": ParameterValue(value=300.0, source="user_provided")},
                outputs={"structure": "structure.pdb"},
            ),
            TaskIR(
                id="run_md",
                name="Run NEMD",
                purpose="Propagate dynamics under field",
                task_type="md_runner",
                inputs={"structure": ParameterValue(value="structure.pdb", source="user_provided")},
                outputs={"trajectory": "traj.dcd"},
            ),
        ],
        edges=[DependencyEdge(source_task_id="build", target_task_id="run_md")],
        expected_outputs=[
            ExpectedOutput(name="trajectory", kind="dataset", description="MD trajectory"),
        ],
    )
    dumped = ir.model_dump_json()
    rehydrated = WorkflowIR.model_validate_json(dumped)
    assert rehydrated == ir


def test_workflow_ir_is_frozen() -> None:
    from molexp.harness.schemas.workflow_ir import WorkflowIR

    ir = WorkflowIR(
        id="wf-001",
        name="x",
        objective="y",
        inputs={},
        tasks=[],
        edges=[],
        expected_outputs=[],
    )
    with pytest.raises(ValidationError):
        ir.name = "mutated"  # type: ignore[misc]


def test_workflow_ir_defaults() -> None:
    from molexp.harness.schemas.workflow_ir import WorkflowIR

    ir = WorkflowIR(
        id="wf-001",
        name="x",
        objective="y",
        inputs={},
        tasks=[],
        edges=[],
        expected_outputs=[],
    )
    assert ir.assumptions == []
    assert ir.review_flags == []


def test_task_ir_round_trip() -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import TaskIR

    task = TaskIR(
        id="t1",
        name="Build polymer",
        purpose="Generate atomistic configuration",
        task_type="molecule_builder",
        inputs={"n_chains": ParameterValue(value=100, source="agent_inferred")},
        outputs={"structure": "structure.pdb"},
        constraints={"min_density": ParameterValue(value=0.9, source="package_default")},
        suggested_capabilities=["molpy.builder.polymer.GBigSmilesCompiler"],
        acceptance_criteria=["density within 5% of target"],
        review_flags=["n_chains"],
    )
    dumped = task.model_dump_json()
    rehydrated = TaskIR.model_validate_json(dumped)
    assert rehydrated == task


def test_task_ir_defaults() -> None:

    task = _make_task()
    assert task.constraints == {}
    assert task.suggested_capabilities == []
    assert task.acceptance_criteria == []
    assert task.review_flags == []


def test_task_ir_is_frozen() -> None:

    task = _make_task()
    with pytest.raises(ValidationError):
        task.task_type = "mutated"  # type: ignore[misc]


def test_dependency_edge_round_trip_and_defaults() -> None:
    from molexp.harness.schemas.workflow_ir import DependencyEdge

    edge = DependencyEdge(source_task_id="a", target_task_id="b")
    assert edge.relation == "requires"  # default

    edge2 = DependencyEdge(source_task_id="a", target_task_id="b", relation="repairs")
    assert edge2.relation == "repairs"

    dumped = edge2.model_dump_json()
    rehydrated = DependencyEdge.model_validate_json(dumped)
    assert rehydrated == edge2


def test_dependency_edge_is_frozen() -> None:
    from molexp.harness.schemas.workflow_ir import DependencyEdge

    edge = DependencyEdge(source_task_id="a", target_task_id="b")
    with pytest.raises(ValidationError):
        edge.relation = "mutated"  # type: ignore[misc]


def test_expected_output_round_trip() -> None:
    from molexp.harness.schemas.workflow_ir import ExpectedOutput

    eo = ExpectedOutput(
        name="trajectory",
        kind="dataset",
        description="MD trajectory file",
        required=True,
    )
    dumped = eo.model_dump_json()
    rehydrated = ExpectedOutput.model_validate_json(dumped)
    assert rehydrated == eo
    assert eo.required is True


def test_expected_output_required_default_true() -> None:
    from molexp.harness.schemas.workflow_ir import ExpectedOutput

    eo = ExpectedOutput(name="x", kind="log", description="some log")
    assert eo.required is True


def test_expected_output_kind_uses_artifact_kind_literal() -> None:
    """kind MUST be Phase-1 ArtifactKind, not a free-form string."""
    from molexp.harness.schemas.workflow_ir import ExpectedOutput

    with pytest.raises(ValidationError):
        ExpectedOutput(name="x", kind="not_a_real_kind", description="x")  # type: ignore[arg-type]


def test_expected_output_is_frozen() -> None:
    from molexp.harness.schemas.workflow_ir import ExpectedOutput

    eo = ExpectedOutput(name="x", kind="log", description="x")
    with pytest.raises(ValidationError):
        eo.name = "mutated"  # type: ignore[misc]


def test_workflow_ir_inputs_reject_non_parameter_value() -> None:
    """ParameterValue type is enforced inside the dict."""
    from molexp.harness.schemas.workflow_ir import WorkflowIR

    with pytest.raises(ValidationError):
        WorkflowIR(
            id="wf-001",
            name="x",
            objective="y",
            inputs={"bad": "plain string not allowed"},  # type: ignore[dict-item]
            tasks=[],
            edges=[],
            expected_outputs=[],
        )


def test_task_ir_constraints_reject_non_parameter_value() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR

    with pytest.raises(ValidationError):
        TaskIR(
            id="t1",
            name="x",
            purpose="x",
            task_type="x",
            inputs={},
            outputs={"out": "out.txt"},
            constraints={"bad": 123},  # type: ignore[dict-item]
        )


def test_default_factories_are_independent() -> None:
    """No shared mutable list/dict defaults across instances."""
    from molexp.harness.schemas.workflow_ir import WorkflowIR

    a = WorkflowIR(
        id="a", name="a", objective="a", inputs={}, tasks=[], edges=[], expected_outputs=[]
    )
    b = WorkflowIR(
        id="b", name="b", objective="b", inputs={}, tasks=[], edges=[], expected_outputs=[]
    )
    assert a.assumptions is not b.assumptions
    assert a.review_flags is not b.review_flags

    t1 = _make_task("t1")
    t2 = _make_task("t2")
    assert t1.review_flags is not t2.review_flags
    assert t1.constraints is not t2.constraints
