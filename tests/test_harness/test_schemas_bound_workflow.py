"""Tests for BoundWorkflow / BoundTask / ExecutionEnvironment / ResourcePolicy
(Phase 3 §4.7).

Locks the wire format:
- frozen pydantic round-trip
- Optional fields default to None / empty
- DependencyEdge re-used from workflow_ir (not redefined)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _make_bound_task(*, id_: str = "b1", ir_task_id: str = "t1"):
    from molexp.harness.schemas.bound_workflow import BoundTask
    from molexp.harness.schemas.parameter import ParameterValue

    return BoundTask(
        id=id_,
        ir_task_id=ir_task_id,
        capability_id="molpy.builder.polymer.GBigSmilesCompiler",
        package="molpy",
        callable="molpy.builder.polymer.GBigSmilesCompiler",
        parameters={"n_chains": ParameterValue(value=100, source="user_provided")},
        inputs={},
        outputs={"structure": "structure.pdb"},
    )


def test_bound_task_round_trip() -> None:
    from molexp.harness.schemas.bound_workflow import BoundTask

    bt = _make_bound_task()
    dumped = bt.model_dump_json()
    rehydrated = BoundTask.model_validate_json(dumped)
    assert rehydrated == bt


def test_bound_task_defaults() -> None:
    bt = _make_bound_task()
    assert bt.version is None
    assert bt.command_template is None
    assert bt.side_effects == []
    assert bt.tests == []
    assert bt.provenance == {}


def test_bound_task_is_frozen() -> None:
    bt = _make_bound_task()
    with pytest.raises(ValidationError):
        bt.capability_id = "mutated"  # type: ignore[misc]


def test_execution_environment_round_trip_all_none() -> None:
    from molexp.harness.schemas.bound_workflow import ExecutionEnvironment

    env = ExecutionEnvironment()
    assert env.python_version is None
    assert env.packages == {}
    assert env.git_commit is None
    assert env.container_image is None
    assert env.env_vars == {}
    assert env.platform is None
    dumped = env.model_dump_json()
    rehydrated = ExecutionEnvironment.model_validate_json(dumped)
    assert rehydrated == env


def test_execution_environment_round_trip_populated() -> None:
    from molexp.harness.schemas.bound_workflow import ExecutionEnvironment

    env = ExecutionEnvironment(
        python_version="3.12.12",
        packages={"molpy": "0.1.0", "numpy": "1.26"},
        git_commit="abc123",
        container_image="ghcr.io/molcrafts/molpy:0.1.0",
        env_vars={"MOLEXP_HOME": "/tmp/m"},
        platform="darwin-arm64",
    )
    dumped = env.model_dump_json()
    rehydrated = ExecutionEnvironment.model_validate_json(dumped)
    assert rehydrated == env


def test_execution_environment_is_frozen() -> None:
    from molexp.harness.schemas.bound_workflow import ExecutionEnvironment

    env = ExecutionEnvironment(python_version="3.12.12")
    with pytest.raises(ValidationError):
        env.python_version = "mutated"  # type: ignore[misc]


def test_resource_policy_round_trip_and_defaults() -> None:
    from molexp.harness.schemas.bound_workflow import ResourcePolicy

    rp = ResourcePolicy(backend="local", max_runtime_s=3600)
    assert rp.max_memory_gb is None
    assert rp.max_gpu_count is None
    assert rp.allowed_paths == []
    assert rp.denied_paths == []
    assert rp.allow_network is False
    dumped = rp.model_dump_json()
    rehydrated = ResourcePolicy.model_validate_json(dumped)
    assert rehydrated == rp


def test_resource_policy_is_frozen() -> None:
    from molexp.harness.schemas.bound_workflow import ResourcePolicy

    rp = ResourcePolicy(backend="local", max_runtime_s=3600)
    with pytest.raises(ValidationError):
        rp.allow_network = True  # type: ignore[misc]


def test_bound_workflow_round_trip() -> None:
    from molexp.harness.schemas.bound_workflow import (
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.workflow_ir import DependencyEdge

    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[_make_bound_task()],
        edges=[DependencyEdge(source_task_id="b1", target_task_id="b1")],
        execution_backend="local",
        environment=ExecutionEnvironment(python_version="3.12"),
        resource_policy=ResourcePolicy(backend="local", max_runtime_s=3600),
    )
    dumped = bw.model_dump_json()
    rehydrated = BoundWorkflow.model_validate_json(dumped)
    assert rehydrated == bw


def test_bound_workflow_edges_uses_dependency_edge_from_workflow_ir() -> None:
    """Same DependencyEdge type as WorkflowIR — not re-defined."""
    from molexp.harness.schemas.bound_workflow import BoundWorkflow
    from molexp.harness.schemas.workflow_ir import DependencyEdge as IREdge

    field = BoundWorkflow.model_fields["edges"]
    # The annotation is list[DependencyEdge]; type origin = list, args[0] = IREdge.
    from typing import get_args

    edge_type = get_args(field.annotation)[0]
    assert edge_type is IREdge


def test_bound_workflow_is_frozen() -> None:
    from molexp.harness.schemas.bound_workflow import (
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )

    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[],
        edges=[],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(backend="local", max_runtime_s=3600),
    )
    with pytest.raises(ValidationError):
        bw.execution_backend = "mutated"  # type: ignore[misc]


def test_bound_workflow_defaults() -> None:
    from molexp.harness.schemas.bound_workflow import (
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )

    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[],
        edges=[],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(backend="local", max_runtime_s=3600),
    )
    assert bw.review_flags == []
