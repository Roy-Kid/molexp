"""Checker cluster — the three pure conformance checks ReviewMode runs.

Each checker consumes the shared ``molexp.agent.modes._planning``
contracts verbatim and returns a tuple of
:class:`~molexp.agent.modes.review.verdict.StepFinding`\\ s. The three
checks are orthogonal:

- :func:`check_intent_conformance` — every :class:`IntentSpec` success
  criterion / required output is still covered, no ``non_goal`` is
  pursued, and no side effect exceeds ``allowed_side_effects``.
- :func:`check_capability_evidence` — every :class:`PlanStep` carries
  non-empty ``api_refs`` (the planner's MCP-grounded primitive list).
- :func:`check_lifecycle_consistency` — every blocking
  :class:`PlanCheck` is structurally backed and the :class:`PlanState`
  is consistent with what the plan actually contains.

Pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    IntentSpec,
    PlanGraph,
    PlanState,
    PlanStep,
)
from molexp.agent.modes.review.verdict import StepFinding

__all__ = [
    "check_capability_evidence",
    "check_intent_conformance",
    "check_lifecycle_consistency",
]

_INTENT_REQUIRED_OUTPUT = "intent_required_output_covered"
_INTENT_SUCCESS_CRITERION = "intent_success_criterion_covered"
_INTENT_NON_GOAL = "intent_non_goal_respected"
_CAPABILITY_EVIDENCED = "plan_step_capability_evidenced"
_LIFECYCLE_CHECK_BACKED = "lifecycle_check_backed"

# Plan states that assert the plan is ready to run / has run.
_RUN_READY_STATES: frozenset[PlanState] = frozenset(
    {PlanState.ready_for_run, PlanState.running, PlanState.completed}
)


# ── check 1 — intent conformance ─────────────────────────────────────────


def _planned_artifact_paths(plan: PlanGraph) -> set[str]:
    """Collect every artefact path the plan's steps are expected to produce."""
    return {artifact.path for step in plan.steps for artifact in step.artifacts}


def _terminal_step_id(plan: PlanGraph) -> str:
    """Return the last plan step's ``id`` — the natural attach point.

    A missing required output has no producing step, so a repair diff
    attaches to the plan's terminal step (the one no other step depends
    on, falling back to the last in topological order). Returns ``""``
    for an empty plan.
    """
    if not plan.steps:
        return ""
    depended_on = {dep for step in plan.steps for dep in step.depends_on}
    leaves = [step.id for step in plan.steps if step.id not in depended_on]
    return leaves[-1] if leaves else plan.steps[-1].id


def check_intent_conformance(intent: IntentSpec, plan: PlanGraph) -> tuple[StepFinding, ...]:
    """Check the plan still satisfies the :class:`IntentSpec` contract.

    Emits an ``error`` finding for every ``required_output`` no plan step
    produces, an ``error`` for every verifiable ``success_criterion`` the
    plan covers with no artefact, and a ``warning`` for a ``non_goal``
    whose phrasing surfaces in the plan notes. A missing-output finding
    is scoped to the plan's terminal step — the natural place a repair
    re-attaches the dropped artefact.

    Args:
        intent: The :class:`IntentSpec` the plan is judged against.
        plan: The :class:`PlanGraph` under review.

    Returns:
        A tuple of :class:`StepFinding`\\ s — empty when the plan
        conforms.
    """
    findings: list[StepFinding] = []
    produced = _planned_artifact_paths(plan)
    terminal = _terminal_step_id(plan)

    for required in intent.required_outputs:
        if required not in produced:
            findings.append(
                StepFinding(
                    step_id=terminal,
                    severity="error",
                    summary=f"required output {required!r} is not produced",
                    detail=(
                        f"IntentSpec lists {required!r} in required_outputs but no "
                        f"plan step declares an artefact at that path."
                    ),
                    failed_invariant=_INTENT_REQUIRED_OUTPUT,
                )
            )

    findings.extend(_success_criteria_findings(intent, produced, terminal))
    findings.extend(_non_goal_findings(intent, plan))
    return tuple(findings)


def _success_criteria_findings(
    intent: IntentSpec, produced: set[str], terminal: str
) -> list[StepFinding]:
    """Flag verifiable success criteria the plan produces no artefact for."""
    if produced:
        return []
    out: list[StepFinding] = []
    for criterion in intent.success_criteria:
        if criterion.verifiable:
            out.append(
                StepFinding(
                    step_id=terminal,
                    severity="error",
                    summary=f"no artefact backs success criterion: {criterion.summary}",
                    detail=(
                        "The plan declares no artefacts, so the verifiable success "
                        f"criterion {criterion.summary!r} cannot be checked."
                    ),
                    failed_invariant=_INTENT_SUCCESS_CRITERION,
                )
            )
    return out


def _non_goal_findings(intent: IntentSpec, plan: PlanGraph) -> list[StepFinding]:
    """Flag steps whose outputs / notes echo a declared ``non_goal``."""
    out: list[StepFinding] = []
    haystack = plan.notes.lower()
    for non_goal in intent.non_goals:
        needle = non_goal.lower().strip()
        if needle and needle in haystack:
            out.append(
                StepFinding(
                    step_id="",
                    severity="warning",
                    summary=f"plan notes mention a declared non-goal: {non_goal!r}",
                    detail=f"IntentSpec.non_goals forbids {non_goal!r}; the plan notes reference it.",
                    failed_invariant=_INTENT_NON_GOAL,
                )
            )
    return out


# ── check 2 — capability evidence ────────────────────────────────────────


def check_capability_evidence(plan: PlanGraph) -> tuple[StepFinding, ...]:
    """Check every :class:`PlanStep` carries non-empty ``api_refs``.

    After the ``plan-mode-pydanticai-rewrite``, capability evidence lives
    inline on each step. A step with empty ``api_refs`` means the
    research-and-plan agent could not ground the step in any toolchain
    primitive — flag it as an ``error`` so the reviewer sees the gap.

    Args:
        plan: The :class:`PlanGraph` under review.

    Returns:
        A tuple of :class:`StepFinding`\\ s — empty when every step has
        at least one ``api_refs`` entry.
    """
    findings: list[StepFinding] = []
    for step in plan.steps:
        if not step.api_refs:
            findings.append(
                StepFinding(
                    step_id=step.id,
                    severity="error",
                    summary=f"step {step.id!r} has no api_refs",
                    detail=(
                        f"Step {step.id!r} carries no api_refs — the research-and-plan "
                        f"agent could not ground this step in any toolchain primitive."
                    ),
                    failed_invariant=_CAPABILITY_EVIDENCED,
                )
            )
    return tuple(findings)


# ── check 3 — lifecycle consistency ──────────────────────────────────────


def check_lifecycle_consistency(plan: PlanGraph) -> tuple[StepFinding, ...]:
    """Check the :class:`PlanState` is consistent with the plan contents.

    A plan claiming a run-ready state (:data:`PlanState.ready_for_run`,
    ``running``, or ``completed``) must have every step's blocking
    :class:`PlanCheck` structurally present. A run-ready step that
    produces artefacts yet declares *no* blocking check is flagged — the
    plan asserts readiness it cannot verify. A cyclic ``depends_on``
    graph is always an ``error``.

    Args:
        plan: The :class:`PlanGraph` under review.

    Returns:
        A tuple of :class:`StepFinding`\\ s — empty when the lifecycle
        is consistent.
    """
    findings: list[StepFinding] = []
    if not plan.is_acyclic():
        findings.append(
            StepFinding(
                step_id="",
                severity="error",
                summary="the plan's depends_on graph contains a cycle",
                detail="A cyclic plan can never reach a runnable state.",
                failed_invariant=_LIFECYCLE_CHECK_BACKED,
            )
        )
    if plan.state in _RUN_READY_STATES:
        findings.extend(_run_ready_step_findings(plan))
    return tuple(findings)


def _run_ready_step_findings(plan: PlanGraph) -> list[StepFinding]:
    """Flag run-ready steps with artefacts but no blocking check."""
    out: list[StepFinding] = []
    for step in plan.steps:
        if step.artifacts and not _has_blocking_check(step):
            out.append(
                StepFinding(
                    step_id=step.id,
                    severity="warning",
                    summary=(f"step {step.id!r} is run-ready but declares no blocking check"),
                    detail=(
                        f"Plan state is {plan.state.value!r}, yet step {step.id!r} "
                        f"produces artefacts with no blocking PlanCheck to verify them."
                    ),
                    failed_invariant=_LIFECYCLE_CHECK_BACKED,
                )
            )
    return out


def _has_blocking_check(step: PlanStep) -> bool:
    """Return whether ``step`` carries at least one blocking :class:`PlanCheck`."""
    return any(check.blocking for check in step.checks)
