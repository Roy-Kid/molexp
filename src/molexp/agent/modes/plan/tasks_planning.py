"""The read-only PlanMode stage logic.

Seven stages turn a user report into an approved typed
:class:`~molexp.agent.modes._planning.PlanGraph`:

1. :func:`synthesize_intent` — LLM projection of free text → ``IntentSpec``.
2. :func:`clarify_intent` — surface blocking ``MissingInfoItem``\\ s.
3. ``ExploreCapabilities`` — probe + project (see ``_mode.py`` /
   ``capability_projection.py``).
4. :func:`synthesize_candidates` — emit one or three candidate plans.
5. :func:`select_plan` — choose one candidate ``PlanGraph``.
6. ``PreflightPlanGraph`` — see ``plan_graph_preflight.py``.
7. ``EmitApprovedPlan`` — see ``_mode.py``.

These are plain async functions, not workflow ``Task``\\ s — PlanMode
runs them as a plain async sequence on the harness. Each LLM call goes
through the :class:`~molexp.agent.router.Router` structured path. The
module imports no pydantic SDK and no ``molexp.workflow``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import (
    CapabilityGraph,
    IntentSpec,
    MissingInfoItem,
    PlanDiff,
    PlanGraph,
    PlanNodeOp,
    PlanState,
)
from molexp.agent.router import ModelTier, Router

__all__ = [
    "CandidateSet",
    "PlanCandidate",
    "SelectionResult",
    "build_repair_diff",
    "clarify_intent",
    "select_plan",
    "synthesize_candidates",
    "synthesize_intent",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ── Stage-4/5 structured schemas ───────────────────────────────────────────


class PlanCandidate(BaseModel):
    """One candidate plan with its self-critique.

    Attributes:
        label: ``"A"`` / ``"B"`` / ``"C"`` — conservative / faster /
            full-production for a complex task; ``"A"`` alone for a
            simple one.
        plan_graph: The candidate typed plan DAG.
        self_critique: The LLM's own critique of the candidate's
            trade-offs.
    """

    model_config = _FROZEN

    label: str
    plan_graph: PlanGraph
    self_critique: str = ""


class CandidateSet(BaseModel):
    """The output of :func:`synthesize_candidates`.

    Attributes:
        candidates: One candidate for a simple task; three (A/B/C) for a
            complex one.
        is_complex: Whether the LLM judged the task complex enough to
            warrant three candidates.
    """

    model_config = _FROZEN

    candidates: tuple[PlanCandidate, ...]
    is_complex: bool = False


class SelectionResult(BaseModel):
    """The output of the :func:`select_plan` LLM call.

    Attributes:
        chosen_label: The ``PlanCandidate.label`` to select.
        rationale: Why that candidate was chosen.
    """

    model_config = _FROZEN

    chosen_label: str
    rationale: str = ""


# ── Stage 1 — SynthesizeIntent ─────────────────────────────────────────────


_INTENT_SYSTEM_PROMPT = (
    "You are an intent synthesizer. Project the user's free-text "
    "request into a formal IntentSpec: a one-sentence objective, "
    "explicit non-goals, the artefacts the user must receive "
    "(required_outputs), hard constraints, assumptions, open questions "
    "(missing_information — mark blocking=true only when planning "
    "genuinely cannot proceed without the answer), success criteria, "
    "the side effects the user has sanctioned, a resource budget, and "
    "an overall risk level. Do not invent requirements the user did "
    "not state."
)


async def synthesize_intent(*, router: Router, user_input: str) -> IntentSpec:
    """Stage 1 — project ``user_input`` into a typed :class:`IntentSpec`."""
    return await router.complete_structured(
        tier=ModelTier.HEAVY,
        system=_INTENT_SYSTEM_PROMPT,
        user=user_input,
        schema=IntentSpec,
        node_id="SynthesizeIntent",
    )


# ── Stage 2 — ClarifyIntent ────────────────────────────────────────────────


def clarify_intent(*, intent: IntentSpec) -> tuple[PlanState, tuple[MissingInfoItem, ...]]:
    """Stage 2 — route a plan with blocking missing-info to clarification.

    Returns the next :class:`PlanState` and the blocking items. When the
    intent carries a blocking :class:`MissingInfoItem`, the plan moves
    to :data:`PlanState.needs_clarification` and the pipeline stops
    before capability exploration; otherwise it proceeds to
    :data:`PlanState.exploring`.
    """
    blocking = tuple(item for item in intent.missing_information if item.blocking)
    if blocking:
        return PlanState.needs_clarification, blocking
    return PlanState.exploring, ()


# ── Stage 4 — SynthesizeCandidates ─────────────────────────────────────────


_CANDIDATE_SYSTEM_PROMPT = (
    "You are a plan synthesizer. Given a typed IntentSpec and a typed "
    "CapabilityGraph, produce candidate typed PlanGraphs. For a complex "
    "task emit three candidates — A (conservative), B (faster, weaker "
    "validation), C (full production) — each with a self-critique of "
    "its trade-offs and is_complex=true. For a simple task emit a "
    "single candidate (label 'A') and is_complex=false. Every PlanStep "
    "must bind its capability_id to a CapabilityGraph node id, and the "
    "graph must be closed and acyclic. Write no code — only the typed "
    "plan structure."
)


async def synthesize_candidates(
    *,
    router: Router,
    intent: IntentSpec,
    capabilities: CapabilityGraph,
) -> CandidateSet:
    """Stage 4 — synthesize one or three candidate :class:`PlanGraph`\\ s."""
    user = (
        "IntentSpec:\n"
        f"{intent.model_dump_json(indent=2)}\n\n"
        "CapabilityGraph:\n"
        f"{capabilities.model_dump_json(indent=2)}"
    )
    return await router.complete_structured(
        tier=ModelTier.HEAVY,
        system=_CANDIDATE_SYSTEM_PROMPT,
        user=user,
        schema=CandidateSet,
        node_id="SynthesizeCandidates",
    )


# ── Stage 5 — SelectPlan ───────────────────────────────────────────────────


_SELECT_SYSTEM_PROMPT = (
    "You are a plan selector. Given a set of candidate PlanGraphs, "
    "choose the single best one by label. Favour the candidate whose "
    "steps bind to evidenced capabilities and whose validation is "
    "strongest within the IntentSpec's budget and risk tolerance. "
    "Return the chosen label and a one-sentence rationale."
)


async def select_plan(
    *,
    router: Router,
    candidates: CandidateSet,
    capabilities: CapabilityGraph,
) -> PlanGraph:
    """Stage 5 — choose one candidate :class:`PlanGraph`.

    A single-candidate set short-circuits the LLM call. With multiple
    candidates, the structured router picks one by label; an unknown
    label falls back to the first candidate.
    """
    if not candidates.candidates:
        raise ValueError("select_plan: the candidate set is empty")
    if len(candidates.candidates) == 1:
        return candidates.candidates[0].plan_graph

    del capabilities  # reserved for future capability-aware scoring
    user = "Candidates:\n" + "\n".join(
        f"- {c.label}: {c.self_critique}" for c in candidates.candidates
    )
    selection = await router.complete_structured(
        tier=ModelTier.DEFAULT,
        system=_SELECT_SYSTEM_PROMPT,
        user=user,
        schema=SelectionResult,
        node_id="SelectPlan",
    )
    for candidate in candidates.candidates:
        if candidate.label == selection.chosen_label:
            return candidate.plan_graph
    return candidates.candidates[0].plan_graph


# ── Repair-diff construction ───────────────────────────────────────────────


def build_repair_diff(
    *,
    failed_invariant: str,
    plan_graph: PlanGraph,
    rationale: str,
    operations: tuple[PlanNodeOp, ...] = (),
    reused: tuple[str, ...] = (),
    invalidated: tuple[str, ...] = (),
) -> PlanDiff:
    """Build a :class:`PlanDiff` for the plan-diff-centric repair loop.

    ``affected_nodes`` is derived from the operations' ``node_id``s.
    """
    affected = tuple(dict.fromkeys(op.node_id for op in operations))
    del plan_graph  # only the operations name the affected nodes
    return PlanDiff(
        failed_invariant=failed_invariant,
        affected_nodes=affected,
        operations=operations,
        rationale=rationale,
        reused=reused,
        invalidated=invalidated,
    )
