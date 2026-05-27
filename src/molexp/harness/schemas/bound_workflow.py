"""``BoundWorkflow`` + constituents — the execution-binding layer.

Per ``.claude/notes/harness-goal.md`` §4.7: ``BoundWorkflow`` describes
*how* the experiment will actually run — which Molcrafts capability
implements each ``TaskIR``, what version, what parameters (with
provenance), what command template, what side effects, what backend,
what environment, what resource policy.

Every ``BoundTask.ir_task_id`` references a
:class:`molexp.harness.schemas.workflow_ir.TaskIR.id`; the structural
validator
(:func:`molexp.harness.validators.bound_workflow.validate_bound_workflow`)
ensures the mapping is one-to-one and that input/output keys agree.

Capability-aware checks (does ``capability_id`` exist in the registry?
do the parameters match its input schema? does the backend support it?)
land in Phase 4 alongside ``CapabilityRegistry`` — the call site stays
stable so Phase 4 is an additive edit.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.parameter import ParameterValue
from molexp.harness.schemas.workflow_ir import DependencyEdge

__all__ = [
    "BoundTask",
    "BoundWorkflow",
    "ExecutionEnvironment",
    "ResourcePolicy",
]


class BoundTask(BaseModel):
    """One concretely-bound execution unit, linked back to its ``TaskIR``."""

    model_config = ConfigDict(frozen=True)

    id: str
    ir_task_id: str
    capability_id: str
    package: str
    callable: str
    version: str | None = None
    parameters: dict[str, ParameterValue]
    inputs: dict[str, str]
    outputs: dict[str, str]
    command_template: list[str] | None = None
    side_effects: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)


class ExecutionEnvironment(BaseModel):
    """Snapshot of the host/runtime environment a bound workflow runs in.

    Fields are recorded but not cross-checked against the host in Phase 3
    (runtime concern; the executor will validate at submission time).
    """

    model_config = ConfigDict(frozen=True)

    python_version: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    git_commit: str | None = None
    container_image: str | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    platform: str | None = None


class ResourcePolicy(BaseModel):
    """Caps + path / network allow-lists enforced at execution time.

    ``denied_paths`` carries a hard floor — the Phase-3 validator refuses
    any bound workflow whose ``denied_paths`` lacks ``"/"`` or
    ``"~/.ssh"`` because the harness's audit guarantees rely on no run
    ever reading SSH keys.
    """

    model_config = ConfigDict(frozen=True)

    backend: str
    max_runtime_s: int
    max_memory_gb: float | None = None
    max_gpu_count: int | None = None
    allowed_paths: list[str] = Field(default_factory=list)
    denied_paths: list[str] = Field(default_factory=list)
    allow_network: bool = False


class BoundWorkflow(BaseModel):
    """The execution-binding layer above :class:`WorkflowIR`.

    Phase-3 invariants enforced by
    :func:`molexp.harness.validators.bound_workflow.validate_bound_workflow`:

    - Every ``BoundTask.ir_task_id`` resolves into the referenced IR
    - No two ``BoundTask``s share an ``ir_task_id`` (one-to-one mapping)
    - Each ``BoundTask``'s ``inputs`` / ``outputs`` keys match the IR
      task's ``inputs`` / ``outputs`` keys exactly
    - ``edges`` topology, after id-translation back to IR task ids,
      equals the IR's own edge set
    """

    model_config = ConfigDict(frozen=True)

    id: str
    workflow_ir_id: str
    tasks: list[BoundTask]
    edges: list[DependencyEdge]
    execution_backend: str
    environment: ExecutionEnvironment
    resource_policy: ResourcePolicy
    review_flags: list[str] = Field(default_factory=list)
