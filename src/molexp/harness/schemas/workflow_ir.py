"""``WorkflowIR`` + constituents — the scientific-intent layer.

Per ``.claude/notes/harness-goal.md`` §4.6: ``WorkflowIR`` describes *what*
the experiment wants to compute, in vocabulary the agent and the user
both understand. It is intentionally free of execution details (no shell
commands, no backend identifiers, no package versions) — those live one
layer down in :class:`molexp.harness.schemas.bound_workflow.BoundWorkflow`,
which references the IR via ``BoundTask.ir_task_id``.

All schemas here are frozen pydantic so an agent-proposed IR cannot be
mutated in place by downstream stages; any change produces a new IR
artifact whose ``parent_ids`` references the original.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import ArtifactKind
from molexp.harness.schemas.parameter import ParameterValue

__all__ = [
    "DependencyEdge",
    "ExpectedOutput",
    "TaskIR",
    "WorkflowIR",
]


class DependencyEdge(BaseModel):
    """Directed edge in a workflow task graph.

    Re-used by both :class:`WorkflowIR.edges` (id refers to ``TaskIR.id``)
    and :class:`molexp.harness.schemas.bound_workflow.BoundWorkflow.edges`
    (id refers to ``BoundTask.id`` — note the id-space switch).
    """

    model_config = ConfigDict(frozen=True)

    source_task_id: str
    target_task_id: str
    relation: str = "requires"


class ExpectedOutput(BaseModel):
    """One of the workflow's declared deliverables.

    ``kind`` is the open ``ArtifactKind = str`` alias so expected outputs
    use the same vocabulary as the artifact store; the validator can
    cheaply check that each ``required=True`` expected output has a
    producer task whose ``outputs`` dict carries the matching name.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    kind: Annotated[ArtifactKind, Field(min_length=1)]
    description: str
    required: bool = True


class TaskIR(BaseModel):
    """One unit of scientific work in a :class:`WorkflowIR`.

    Carries *what* the task wants — purpose, type, scientific parameters
    with provenance, suggested capabilities (hints for binding, not
    bindings), acceptance criteria. It does NOT carry shell commands,
    backend identifiers, or package versions; those land in
    :class:`molexp.harness.schemas.bound_workflow.BoundTask` once the
    binding step has selected a concrete capability.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    purpose: str
    task_type: str
    inputs: dict[str, ParameterValue]
    outputs: dict[str, str]
    constraints: dict[str, ParameterValue] = Field(default_factory=dict)
    suggested_capabilities: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    review_flags: list[str] = Field(default_factory=list)


class WorkflowIR(BaseModel):
    """The scientific-intent layer: what the experiment wants to compute.

    A ``WorkflowIR`` is built by an agent (Phase 4's ``ExtractWorkflowIR``
    stage) from an upstream :class:`molexp.harness.schemas.experiment_report.ExperimentReport`
    artifact, then validated by
    :func:`molexp.harness.validators.workflow_ir.validate_workflow_ir`.
    The validator catches structural defects (duplicate ids, dangling
    edges, cycles, missing producers, unresolved inputs) plus a
    defense-in-depth grep for shell commands and backend leaks.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    objective: str
    inputs: dict[str, ParameterValue]
    tasks: list[TaskIR]
    edges: list[DependencyEdge]
    expected_outputs: list[ExpectedOutput]
    assumptions: list[str] = Field(default_factory=list)
    review_flags: list[str] = Field(default_factory=list)
