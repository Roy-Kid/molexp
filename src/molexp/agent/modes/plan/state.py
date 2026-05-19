"""Mutable per-execution runtime state for the PlanMode workflow.

Plan tasks thread mutable repair state through ``ctx.deps`` (the
frozen :class:`PlanDeps` aggregate carries a :class:`PlanRuntimeState`
reference whose contents tasks mutate freely). Two collaborating
types live here: :class:`PlanRuntimeState` for live mutation and
:class:`RepairSignal` as the frozen sentinel a task plants when it
wants the workflow loop to re-enter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict


RepairSource = Literal[
    "step_review",
    "capability_required",
    "unevidenced",
    "pipeline_failed",
]
"""Enumeration of recoverable gates that can plant a
:class:`RepairSignal`. ``"step_review"`` — per-step ReviewPolicy
rejected; ``"capability_required"`` — discovery probe declined;
``"unevidenced"`` — codegen evidence gate fired;
``"pipeline_failed"`` — pipeline aborted before reaching FinalHandoffCheck."""

if TYPE_CHECKING:
    from molexp.agent.modes.plan.context import PlanRepairContext
    from molexp.agent.modes.plan.plan_folder import RepairIterationRecord
    from molexp.agent.review import ReviewDecision


__all__ = ["PlanRuntimeState", "RepairSignal", "RepairSource"]


class RepairSignal(BaseModel):
    """Frozen marker a task plants on :attr:`PlanRuntimeState.repair_signal`
    when it hits a recoverable gate.

    Downstream tasks see the slot is occupied and skip work;
    ``RepairDecide`` consumes the slot, builds a :class:`ReviewDecision`
    from it, and drives the next iteration's prompts.

    Attributes:
        source_step: Class name of the task that planted the signal.
        source_kind: Which gate fired — see :data:`RepairSource`.
        target_steps: Steps the next iteration should focus on
            (telemetry only — the workflow re-runs the whole body).
        target_task_ids: Per-task ids the LLM should focus on
            (telemetry only).
        cascade_downstream: Whether the rejection drags downstream
            steps along; surfaced into the manifest's repair history.
        reason: Short slug describing why the gate fired.
        feedback: Free-form text fed into the next iteration's
            ``repair_context`` and rendered into LLM prompts.
        refs: API references the codegen gate flagged as unevidenced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_step: str
    source_kind: RepairSource
    target_steps: tuple[str, ...] = ()
    target_task_ids: tuple[str, ...] = ()
    cascade_downstream: bool = True
    reason: str = ""
    feedback: str = ""
    refs: tuple[str, ...] = ()


@dataclass
class PlanRuntimeState:
    """Live mutable state shared across plan-workflow iterations.

    Lives on :attr:`PlanDeps.runtime`; persists for one :meth:`PlanMode.run`
    call. Three loci touch it: :class:`PrepareIteration` clears
    :attr:`repair_signal` at body start; pipeline tasks plant a
    :class:`RepairSignal` and return ``None`` when a gate fires;
    :class:`RepairDecide` reads the signal (or the terminal handoff),
    bumps :attr:`iteration`, and returns ``Next("continue")`` /
    ``Next("exit")``.

    Plain :func:`dataclass` rather than a pydantic model — runtime
    containers carrying live mutation stay outside the project's
    frozen-pydantic data surface.

    Attributes:
        iteration: Bumped by ``RepairDecide`` after every rejection.
        repair_signal: Slot tasks plant a :class:`RepairSignal` on.
            ``None`` means "iteration healthy".
        repair_context: Feedback from the previous rejection, threaded
            into LLM prompts.
        latest_decision: Most recent :class:`ReviewDecision` that drove
            a rejection.
        repair_history: Append-only audit trail mirrored to the
            manifest's ``repair_history`` field.
        capability_unevidenced_count: Counts codegen-gate firings so
            the second miss escalates to also re-running
            ``DraftCapabilityNeeds``.
    """

    iteration: int = 0
    repair_signal: RepairSignal | None = None
    repair_context: PlanRepairContext | None = None
    latest_decision: ReviewDecision | None = None
    repair_history: list[RepairIterationRecord] = field(default_factory=list)
    capability_unevidenced_count: int = 0
    # One-shot resume payload: consumed by RunPlanIteration on the
    # first iteration, cleared so later iterations run fresh.
    resume_seed_outputs: dict[str, Any] | None = None
    resume_execution_id: str | None = None
    resume_run_dir: str | None = None
    # Inner-workflow outputs from the most recent iteration; the outer
    # workflow's own outputs only cover the loop tasks themselves, so
    # PlanMode.run reads its result projection from here.
    last_inner_outputs: dict[str, Any] = field(default_factory=dict)
