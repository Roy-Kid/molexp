"""Plan-graph cluster — the typed plan DAG.

``PlanGraph`` is the agent-side typed plan: the single source of truth
read by the UI, the review queue, resume, and the repair loop. It is
*upstream* of the workflow layer's compiled ``WorkflowContract`` —
AuthorMode lowers a ``PlanGraph`` into a ``WorkflowContract`` and records
the result in :attr:`PlanGraph.compiled_contract_ref`. The two are
related by reference only; this module imports no workflow types.

Pure frozen-pydantic data models; no LLM, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .diff import ApprovalGate
from .intent import RiskLevel
from .lifecycle import PlanState


class PlanStepInput(BaseModel):
    """One named input a plan step consumes.

    Attributes:
        name: The input's logical name.
        source_step: ``id`` of the upstream step that produces it, or
            ``None`` when the input is externally supplied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    source_step: str | None


class PlanStepIO(BaseModel):
    """The planning-stage I/O sketch of a plan step.

    Deliberately looser than the workflow layer's ``TaskIO`` — it exists
    before any workflow IR is compiled.

    Attributes:
        inputs: The named inputs the step consumes.
        outputs: The logical names of the values the step produces.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: tuple[PlanStepInput, ...]
    outputs: tuple[str, ...]


class PlanStepArtifact(BaseModel):
    """A file artefact a plan step is expected to produce.

    Attributes:
        path: Workspace-relative path of the artefact.
        description: What the artefact contains.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    description: str


class PlanCheck(BaseModel):
    """A validation check attached to a plan step.

    Attributes:
        name: Short identifier of the check.
        description: What the check verifies.
        blocking: Whether a failure blocks the plan from proceeding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    blocking: bool


class RetryPolicy(BaseModel):
    """The retry policy of a plan step.

    Attributes:
        max_attempts: Total attempts allowed, ``>= 1`` (default ``1``).
        on: Failure-kind tags the step retries on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=1, ge=1)
    on: tuple[str, ...]


class IsolatedTestSketch(BaseModel):
    """A testability verdict for one plan step.

    Records whether a plan step is small enough to carry an independent,
    low-cost isolated test — the termination criterion of
    testability-driven decomposition. A step that still needs the real
    output of an upstream step to be exercised is not isolated-testable
    and must be split further before the plan can pass preflight.

    Attributes:
        is_isolated_testable: Whether a standalone, low-cost test exists
            for the step. Decomposition stops splitting a branch once
            every step on it is ``True``.
        synthetic_inputs: Logical descriptions of the minimal synthetic
            inputs the isolated test supplies itself — never the real
            output of an upstream step.
        assertion_sketch: What the isolated test should assert.
        rationale: When ``is_isolated_testable`` is ``False``, why the
            step cannot yet be tested in isolation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_isolated_testable: bool
    synthetic_inputs: tuple[str, ...]
    assertion_sketch: tuple[str, ...]
    rationale: str


class PlanStep(BaseModel):
    """One node of the typed plan DAG.

    Attributes:
        id: Stable step identifier.
        depends_on: ``id``s of the steps this step depends on.
        io: The planning-stage I/O sketch.
        artifacts: Artefacts the step is expected to produce.
        capability_id: ``id`` of the bound ``CapabilityNode``, or ``None``.
        tool_binding: A bound tool identifier, or ``None``.
        checks: Validation checks attached to the step.
        retry_policy: The step's retry policy.
        rollback: A rollback description, or ``None``.
        approval_gate: The human approval gate the step sits behind.
        estimated_cost_usd: Estimated cost in USD, or ``None``.
        risk_level: Risk classification of the step.
        unknowns: Open unknowns recorded against the step.
        test_sketch: The step's testability verdict — the
            decomposition's per-step termination record. Required; a
            plan step always carries one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    depends_on: tuple[str, ...]
    io: PlanStepIO
    artifacts: tuple[PlanStepArtifact, ...]
    capability_id: str | None
    tool_binding: str | None
    checks: tuple[PlanCheck, ...]
    retry_policy: RetryPolicy
    rollback: str | None
    approval_gate: ApprovalGate
    estimated_cost_usd: float | None
    risk_level: RiskLevel
    unknowns: tuple[str, ...]
    test_sketch: IsolatedTestSketch


class PlanGraph(BaseModel):
    """The typed plan DAG — the agent-side single source of truth.

    Attributes:
        plan_id: Stable plan identifier.
        intent_ref: Identifier of the originating ``IntentSpec``, or
            ``None``.
        steps: The plan steps, in topological order.
        state: The plan's machine-readiness lifecycle state.
        compiled_contract_ref: ``workflow_id`` of the ``WorkflowContract``
            AuthorMode lowered this plan into, or ``None`` before
            materialization.
        notes: Free-form notes about the plan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    intent_ref: str | None
    steps: tuple[PlanStep, ...]
    state: PlanState
    compiled_contract_ref: str | None
    notes: str

    def step_by_id(self, step_id: str) -> PlanStep | None:
        """Return the step with ``id == step_id``, or ``None``."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def downstream_of(self, step_id: str) -> tuple[str, ...]:
        """Return the transitive dependents of ``step_id``, in plan order."""
        reachable: set[str] = set()
        changed = True
        while changed:
            changed = False
            for step in self.steps:
                if step.id == step_id or step.id in reachable:
                    continue
                if any(dep == step_id or dep in reachable for dep in step.depends_on):
                    reachable.add(step.id)
                    changed = True
        return tuple(step.id for step in self.steps if step.id in reachable)

    def is_acyclic(self) -> bool:
        """Return whether the ``depends_on`` graph has no cycle."""
        ids = {step.id for step in self.steps}
        dependents: dict[str, list[str]] = {sid: [] for sid in ids}
        indegree: dict[str, int] = dict.fromkeys(ids, 0)
        for step in self.steps:
            for dep in step.depends_on:
                if dep in ids:
                    dependents[dep].append(step.id)
                    indegree[step.id] += 1
        queue = [sid for sid in ids if indegree[sid] == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for dependent in dependents[node]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        return visited == len(ids)
