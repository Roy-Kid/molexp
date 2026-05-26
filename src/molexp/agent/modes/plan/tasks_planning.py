"""Read-only PlanMode stage logic — the two LLM-free or single-LLM steps.

After ``plan-mode-pydanticai-rewrite`` collapsed the seven-stage pipeline
into five (one structured + one MCP-attached agentic call), only two
stage helpers remain in this module:

1. :func:`synthesize_intent` — LLM projection of free text → ``IntentSpec``
   (one structured call through the :class:`~molexp.agent.router.Router`).
2. :func:`clarify_intent` — pure function: route to
   ``needs_clarification`` if the intent carries a blocking missing-info
   item.

The previous N-call orchestration —
``synthesize_candidates`` / ``refine_until_testable`` / ``select_plan``
plus the ``CandidateSet`` / ``PlanCandidate`` / ``SelectionResult`` /
``StepRefinement`` schemas — moved into a single pydantic-ai-native
research-and-plan agent in
:mod:`molexp.agent._pydanticai.research_planner`. The
:class:`~molexp.agent.modes.plan.stages.research_and_plan.ResearchAndPlan`
stage drives that agent end-to-end; the LLM decides how many tool calls
and how much decomposition the plan needs, inside one
``agent.run(prompt)``.

The :func:`build_repair_diff` helper survives — it still constructs a
:class:`PlanDiff` for the rejection-driven repair path, which now also
rewinds to ``ResearchAndPlan`` rather than the deleted
``SynthesizeCandidates``.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    IntentSpec,
    MissingInfoItem,
    PlanDiff,
    PlanGraph,
    PlanNodeOp,
    PlanState,
)
from molexp.agent.router import ModelTier, Router

__all__ = [
    "build_repair_diff",
    "clarify_intent",
    "synthesize_intent",
]


# ── Stage 1 — SynthesizeIntent ─────────────────────────────────────────────


_INTENT_SYSTEM_PROMPT = (
    "You are an intent synthesizer. Project the user's free-text "
    "request into a formal IntentSpec: a one-sentence objective, "
    "explicit non-goals, the artefacts the user must receive "
    "(required_outputs), hard constraints, assumptions, open questions "
    "(missing_information), success criteria, the side effects the user "
    "has sanctioned, a resource budget, and an overall risk level. Do "
    "not invent requirements the user did not state.\n"
    "\n"
    "MISSING-INFORMATION RULES — read carefully:\n"
    "  - Mark `blocking=true` ONLY when planning genuinely cannot "
    "proceed without a human answer (e.g. an undefined target "
    "molecule, a missing input file, contradictory requirements).\n"
    "  - DO NOT mark API-discovery as blocking. The downstream "
    "ResearchAndPlan stage has an MCP toolset attached to the project "
    "source — discovering 'which symbols implement this operation' is "
    "its job, not the user's. Questions like 'what APIs exist for X?' "
    "are NOT blocking missing-info.\n"
    "  - DO NOT mark library-availability as blocking. Assume the "
    "project toolchain attached to the MCP server is installed and "
    "importable unless the user explicitly says otherwise.\n"
    "  - Implementation choices the user did not pin (e.g. 'which "
    "thermostat?', 'which integrator?') are `blocking=false` — record "
    "them so the planner can pick a reasonable default.\n"
    "\n"
    "Each required_outputs entry MUST be a short identifier-like "
    "artefact name — e.g. 'trajectory', 'data.peo', 'tg_report.csv' — "
    "never a prose sentence or description. Downstream the plan's step "
    "outputs are matched against these strings verbatim, so keep them "
    "terse, lowercase, and free of explanatory text."
)


async def synthesize_intent(*, router: Router, user_input: str) -> IntentSpec:
    """Stage 1 — project ``user_input`` into a typed :class:`IntentSpec`."""
    return await router.complete_structured(
        tier=ModelTier.DEFAULT,
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
    before research-and-plan; otherwise it proceeds to
    :data:`PlanState.exploring`.
    """
    blocking = tuple(item for item in intent.missing_information if item.blocking)
    if blocking:
        return PlanState.needs_clarification, blocking
    return PlanState.exploring, ()


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
