"""Structured payload types for the PlanMode workflow.

Every task in :mod:`~molexp.agent.modes.plan.tasks` returns one of
these models — string-only data flow is deliberately rejected
(planning needs structure, not natural language hand-off). Models are
considered private to ``modes.plan`` for now; if downstream code grows
a need to consume them they can be elevated to a public namespace.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(frozen=True)


# ── Specification stages ───────────────────────────────────────────────────


class IntakeSpec(BaseModel):
    """Parsed user request — what is being asked, under what constraints."""

    model_config = _FROZEN

    request: str
    extracted_goal: str
    constraints: tuple[str, ...] = ()


class GoalSpec(BaseModel):
    """Refined, single-objective restatement of the request."""

    model_config = _FROZEN

    objective: str
    success_criteria: tuple[str, ...] = ()


class ContextSpec(BaseModel):
    """Constraints / assumptions / environment for the goal."""

    model_config = _FROZEN

    constraints: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    environment: str = ""


class MethodSpec(BaseModel):
    """Chosen experimental / computational method."""

    model_config = _FROZEN

    name: str
    rationale: str = ""


class Decomposition(BaseModel):
    """Ordered protocol stages the method breaks into."""

    model_config = _FROZEN

    stages: tuple[str, ...]


class ProtocolStep(BaseModel):
    """One concrete stage of the protocol — inputs / op / outputs."""

    model_config = _FROZEN

    stage: str
    operation: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


class ProtocolDraft(BaseModel):
    """Full protocol — list of concrete steps."""

    model_config = _FROZEN

    steps: tuple[ProtocolStep, ...]


class PlanSpec(BaseModel):
    """Composed plan — frozen view, materialised inline by Preview / Codegen.

    There is **no** ``ComposePlanTask`` in the workflow; each consumer
    builds a fresh :class:`PlanSpec` from the same upstream
    ``ctx.inputs`` map via the private :func:`compose_plan_spec`
    helper. The shape lives here so producers and consumers agree.
    """

    model_config = _FROZEN

    goal: GoalSpec
    context: ContextSpec
    method: MethodSpec
    decomposition: Decomposition
    protocol: ProtocolDraft
    revision: int = 0


# ── Preview / approvals ────────────────────────────────────────────────────


class PlanPreview(BaseModel):
    """Human-readable rendering of a :class:`PlanSpec`."""

    model_config = _FROZEN

    plan: PlanSpec
    rendered: str


class ApprovalDecision(BaseModel):
    """Output of a gate policy invocation."""

    model_config = _FROZEN

    approved: bool
    note: str = ""


# ── Codegen / executable draft ─────────────────────────────────────────────


class GeneratedTaskSpec(BaseModel):
    """One LLM-authored Task implementation — its own artifact unit."""

    model_config = _FROZEN

    stage: str
    task_id: str
    code: str
    docstring: str = ""


class CodegenOutput(BaseModel):
    """LLM-side codegen output — N independent generated tasks.

    Internal to :class:`~molexp.agent.modes.plan.tasks.CodegenTask`;
    the workflow surface emits :class:`ExecutableWorkflowDraft`.
    """

    model_config = _FROZEN

    generated: tuple[GeneratedTaskSpec, ...]


class ExecutableWorkflowDraft(BaseModel):
    """Plan + bound + generated — what the compiler turns into a template."""

    model_config = _FROZEN

    plan: PlanSpec
    bound: dict[str, str] = Field(default_factory=dict)
    generated: tuple[GeneratedTaskSpec, ...] = ()


# ── Compile / dry-run reports ──────────────────────────────────────────────


class CompileReport(BaseModel):
    model_config = _FROZEN

    ok: bool
    workflow_template_id: str | None = None
    experiment_spec_id: str | None = None
    diagnostics: tuple[str, ...] = ()


class DryRunReport(BaseModel):
    model_config = _FROZEN

    ok: bool
    notes: tuple[str, ...] = ()


# ── Repair ─────────────────────────────────────────────────────────────────


class PlanPatch(BaseModel):
    """Atomic change targeting one node of a :class:`PlanSpec`."""

    model_config = _FROZEN

    target: str
    new_value: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class RepairReport(BaseModel):
    """One repair iteration — patches plus the stale-subgraph it implies.

    ``affected_nodes`` are the spec nodes the patches mutate;
    ``stale_nodes`` are the downstream nodes whose outputs are no longer
    valid. molexp.workflow currently re-enters the plan-mode cycle from
    ``preview`` regardless; recording the markers here keeps the
    contract ready for a future scheduler that re-runs only the
    invalidated subgraph.
    """

    model_config = _FROZEN

    iteration: int
    patches: tuple[PlanPatch, ...] = ()
    affected_nodes: tuple[str, ...] = ()
    stale_nodes: tuple[str, ...] = ()


# ── Final handoff ──────────────────────────────────────────────────────────


class ApprovedPlan(BaseModel):
    """Terminal payload — handed to the runner downstream of PlanMode."""

    model_config = _FROZEN

    plan: PlanSpec
    executable: ExecutableWorkflowDraft
    compile_report: CompileReport
    dry_run_report: DryRunReport
    iterations: int


__all__ = [
    "ApprovalDecision",
    "ApprovedPlan",
    "CodegenOutput",
    "CompileReport",
    "ContextSpec",
    "Decomposition",
    "DryRunReport",
    "ExecutableWorkflowDraft",
    "GeneratedTaskSpec",
    "GoalSpec",
    "IntakeSpec",
    "MethodSpec",
    "PlanPatch",
    "PlanPreview",
    "PlanSpec",
    "ProtocolDraft",
    "ProtocolStep",
    "RepairReport",
]
