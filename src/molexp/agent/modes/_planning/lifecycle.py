"""Lifecycle cluster — the explicit ``PlanState`` machine.

A single machine-readiness lifecycle. Human approval is *not* a state
here — it is expressed separately by which ``ApprovalGate`` a plan step
sits behind; ``awaiting_approval`` is the one state where the machine is
blocked on a human. :data:`LEGAL_TRANSITIONS` is the source of truth for
which moves are allowed.

Pure data + pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from enum import StrEnum


class PlanState(StrEnum):
    """Machine-readiness lifecycle state of a plan.

    The terminal states are ``completed``, ``rejected``, and ``failed``.
    ``ready_for_run`` is the hand-off point AuthorMode reaches and RunMode
    enters at; ``running`` is the in-flight execution state RunMode owns;
    ``completed`` is the terminal-success state.
    """

    intake = "intake"
    needs_clarification = "needs_clarification"
    exploring = "exploring"
    draft_plan = "draft_plan"
    preflight_failed = "preflight_failed"
    awaiting_approval = "awaiting_approval"
    approved = "approved"
    materializing = "materializing"
    validating = "validating"
    ready_for_run = "ready_for_run"
    running = "running"
    completed = "completed"
    rejected = "rejected"
    failed = "failed"


class IllegalPlanTransitionError(ValueError):
    """Raised when a ``PlanState`` move is absent from ``LEGAL_TRANSITIONS``."""


LEGAL_TRANSITIONS: dict[PlanState, frozenset[PlanState]] = {
    PlanState.intake: frozenset({PlanState.needs_clarification, PlanState.exploring}),
    PlanState.needs_clarification: frozenset({PlanState.exploring, PlanState.intake}),
    PlanState.exploring: frozenset({PlanState.draft_plan, PlanState.needs_clarification}),
    PlanState.draft_plan: frozenset({PlanState.preflight_failed, PlanState.awaiting_approval}),
    PlanState.preflight_failed: frozenset(
        {PlanState.exploring, PlanState.draft_plan, PlanState.failed}
    ),
    PlanState.awaiting_approval: frozenset(
        {PlanState.approved, PlanState.rejected, PlanState.draft_plan}
    ),
    PlanState.approved: frozenset({PlanState.materializing}),
    PlanState.materializing: frozenset({PlanState.validating, PlanState.failed}),
    PlanState.validating: frozenset(
        {PlanState.ready_for_run, PlanState.draft_plan, PlanState.failed}
    ),
    PlanState.ready_for_run: frozenset({PlanState.running}),
    PlanState.running: frozenset(
        {PlanState.completed, PlanState.failed, PlanState.needs_clarification}
    ),
    PlanState.completed: frozenset(),
    PlanState.rejected: frozenset(),
    PlanState.failed: frozenset(),
}


def legal_successors(src: PlanState) -> frozenset[PlanState]:
    """Return the set of states legally reachable in one move from ``src``."""
    return LEGAL_TRANSITIONS[src]


def is_legal_transition(src: PlanState, dst: PlanState) -> bool:
    """Return whether moving from ``src`` to ``dst`` is legal."""
    return dst in LEGAL_TRANSITIONS[src]


def assert_legal_transition(src: PlanState, dst: PlanState) -> None:
    """Raise :class:`IllegalPlanTransitionError` if ``src -> dst`` is illegal."""
    if not is_legal_transition(src, dst):
        raise IllegalPlanTransitionError(
            f"illegal plan-state transition: {src.value} -> {dst.value}"
        )
