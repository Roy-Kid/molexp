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
    PlanStep,
)
from molexp.agent.router import ModelTier, Router

__all__ = [
    "CandidateSet",
    "PlanCandidate",
    "SelectionResult",
    "StepRefinement",
    "build_repair_diff",
    "clarify_intent",
    "refine_until_testable",
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


class StepRefinement(BaseModel):
    """The structured output of one ``RefineUntilTestable`` call.

    Attributes:
        sub_steps: The finer-grained ordered replacement for one coarse
            :class:`~molexp.agent.modes._planning.PlanStep`. The first
            sub-step inherits the coarse step's external ``depends_on``;
            the last is the terminal sub-step that downstream steps
            re-point to. Each sub-step carries its own
            :class:`~molexp.agent.modes._planning.IsolatedTestSketch`.
    """

    model_config = _FROZEN

    sub_steps: tuple[PlanStep, ...]


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
    "not state. "
    "Each required_outputs entry MUST be a short identifier-like "
    "artefact name — e.g. 'trajectory', 'data.peo', 'tg_report.csv' — "
    "never a prose sentence or description. Downstream the plan's step "
    "outputs are matched against these strings verbatim, so keep them "
    "terse, lowercase, and free of explanatory text."
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
    "graph must be closed and acyclic. "
    "Artefact vocabulary is load-bearing and checked verbatim: "
    "(1) every entry in IntentSpec.required_outputs must appear, "
    "spelled identically, as an `outputs` entry of some PlanStep; "
    "(2) every step output that is NOT a required output must be "
    "consumed by a downstream step — that step lists it as an input "
    "whose `source_step` is the producer's id and whose `name` is the "
    "exact output string. Reuse output names verbatim across steps; "
    "do not paraphrase. "
    "Every PlanStep must carry a test_sketch: decompose the plan finely "
    "enough that each step is isolated-testable "
    "(test_sketch.is_isolated_testable=true, with concrete "
    "synthetic_inputs and assertion_sketch). A step that could only be "
    "exercised with the real output of an upstream step is too coarse "
    "and must be split further. "
    "Write no code — only the typed plan structure."
)


async def synthesize_candidates(
    *,
    router: Router,
    intent: IntentSpec,
    capabilities: CapabilityGraph,
) -> CandidateSet:
    """Stage 4 — synthesize candidates, then refine each until testable.

    The structured call emits one or three candidate
    :class:`PlanGraph`\\ s; every candidate is then run through
    :func:`refine_until_testable` so its steps are decomposed to
    isolated-testable granularity before selection.
    """
    user = (
        "IntentSpec:\n"
        f"{intent.model_dump_json(indent=2)}\n\n"
        "CapabilityGraph:\n"
        f"{capabilities.model_dump_json(indent=2)}"
    )
    raw = await router.complete_structured(
        tier=ModelTier.HEAVY,
        system=_CANDIDATE_SYSTEM_PROMPT,
        user=user,
        schema=CandidateSet,
        node_id="SynthesizeCandidates",
    )
    refined: list[PlanCandidate] = []
    for candidate in raw.candidates:
        plan_graph = await refine_until_testable(
            router=router, plan_graph=candidate.plan_graph, intent=intent
        )
        refined.append(candidate.model_copy(update={"plan_graph": plan_graph}))
    return raw.model_copy(update={"candidates": tuple(refined)})


# ── Stage 4b — RefineUntilTestable ─────────────────────────────────────────


_MAX_REFINE_DEPTH = 8
"""Maximum step-split rounds for :func:`refine_until_testable` — one
coarse step per round. A plan still carrying a non-isolated-testable
step after this budget is returned as-is; the plan-graph preflight then
fails it closed."""


_REFINE_SYSTEM_PROMPT = (
    "You are a plan-refinement agent. You are given exactly ONE PlanStep "
    "that is too coarse to test in isolation. Split it into a small "
    "ordered chain of finer PlanSteps, each of which IS isolated-testable "
    "— set test_sketch.is_isolated_testable=true with concrete "
    "synthetic_inputs and assertion_sketch. Preserve the original step's "
    "external contract: the first sub-step inherits the original's "
    "depends_on, the last sub-step reproduces the original's io.outputs, "
    "and the sub-steps' internal depends_on form an acyclic chain. Reuse "
    "output names verbatim. Write no code — only the typed sub-steps."
)


async def refine_until_testable(
    *,
    router: Router,
    plan_graph: PlanGraph,
    intent: IntentSpec,
) -> PlanGraph:
    """Recursively split coarse steps until every step is isolated-testable.

    Each round finds the first step whose
    ``test_sketch.is_isolated_testable`` is ``False``, asks the router
    for a finer-grained :class:`StepRefinement`, and replaces the coarse
    step with the sub-steps — re-pointing ``depends_on`` so the graph
    stays closed. Bounded by :data:`_MAX_REFINE_DEPTH`: a plan still
    carrying a non-testable step after the budget is returned as-is for
    the plan-graph preflight to fail it closed. A split that would make
    the graph cyclic is discarded and refinement stops.
    """
    graph = plan_graph
    for _ in range(_MAX_REFINE_DEPTH):
        coarse = _first_coarse_step(graph)
        if coarse is None:
            return graph
        refinement = await router.complete_structured(
            tier=ModelTier.HEAVY,
            system=_REFINE_SYSTEM_PROMPT,
            user=_refine_user_prompt(coarse, intent),
            schema=StepRefinement,
            node_id="RefineUntilTestable",
        )
        candidate = _apply_refinement(graph, coarse, refinement.sub_steps)
        if not candidate.is_acyclic():
            return graph
        graph = candidate
    return graph


def _first_coarse_step(graph: PlanGraph) -> PlanStep | None:
    """Return the first step that is not isolated-testable, or ``None``."""
    for step in graph.steps:
        if not step.test_sketch.is_isolated_testable:
            return step
    return None


def _refine_user_prompt(step: PlanStep, intent: IntentSpec) -> str:
    """Build the user prompt for one ``RefineUntilTestable`` call."""
    return (
        "IntentSpec:\n"
        f"{intent.model_dump_json(indent=2)}\n\n"
        "Coarse PlanStep to split:\n"
        f"{step.model_dump_json(indent=2)}"
    )


def _apply_refinement(
    graph: PlanGraph,
    coarse: PlanStep,
    sub_steps: tuple[PlanStep, ...],
) -> PlanGraph:
    """Replace ``coarse`` with ``sub_steps``, re-pointing ``depends_on``.

    The first sub-step inherits the coarse step's external
    ``depends_on``; every step that depended on the coarse step is
    re-pointed to the terminal (last) sub-step. An empty ``sub_steps``
    leaves the graph unchanged.
    """
    if not sub_steps:
        return graph
    head = sub_steps[0].model_copy(update={"depends_on": coarse.depends_on})
    chain = (head, *sub_steps[1:])
    terminal_id = chain[-1].id
    new_steps: list[PlanStep] = []
    for step in graph.steps:
        if step.id == coarse.id:
            new_steps.extend(chain)
        elif coarse.id in step.depends_on:
            rewired = tuple(terminal_id if dep == coarse.id else dep for dep in step.depends_on)
            new_steps.append(step.model_copy(update={"depends_on": rewired}))
        else:
            new_steps.append(step)
    return graph.model_copy(update={"steps": tuple(new_steps)})


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
