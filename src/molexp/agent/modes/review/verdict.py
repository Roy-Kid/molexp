"""Verdict cluster — the typed ReviewMode output.

:class:`StepFinding` is one observation about one :class:`PlanStep`;
:class:`ReviewVerdict` folds the findings into an ``overall`` outcome
(``pass`` / ``fail`` / ``needs_changes``) and, when changes are needed,
carries a :class:`~molexp.agent.modes._planning.PlanDiff` so the verdict
feeds straight back into the shared plan-diff repair loop the other
modes use.

:func:`build_review_verdict` is the pure fold. A finding carrying a
``failed_invariant`` is *actionable* — it can drive a repair. An
``error`` / ``warning`` finding that is actionable lifts the outcome to
``needs_changes`` and feeds the synthesized ``PlanDiff``; an ``error``
finding with no actionable cause is a hard ``fail`` (no repair is
derivable); an all-``info`` (or empty) finding set is ``pass``.

Pure frozen-pydantic data + a pure function; no LLM, no I/O.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import (
    DiffOpKind,
    IntentSpec,
    PlanDiff,
    PlanGraph,
    PlanNodeOp,
)
from molexp.agent.modes.review.target import ReviewTargetKind

__all__ = ["ReviewVerdict", "StepFinding", "build_review_verdict"]

Severity = Literal["info", "warning", "error"]
"""The severity of one :class:`StepFinding`."""

Outcome = Literal["pass", "fail", "needs_changes"]
"""The overall outcome of a :class:`ReviewVerdict`."""


class StepFinding(BaseModel):
    """One review observation, scoped to a single plan step.

    Attributes:
        step_id: ``id`` of the :class:`PlanStep` the finding concerns, or
            ``""`` when the finding is plan-wide.
        severity: The finding's severity — ``info`` / ``warning`` /
            ``error``.
        summary: A one-line statement of the finding.
        detail: A longer human-readable explanation.
        failed_invariant: The named invariant the finding violated, or
            ``None`` when the finding is purely informational.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str = ""
    severity: Severity
    summary: str
    detail: str = ""
    failed_invariant: str | None = None


class ReviewVerdict(BaseModel):
    """The typed ReviewMode output — per-step findings plus an outcome.

    Carried in ``AgentRunResult.mode_state["verdict"]`` and persisted by
    :class:`~molexp.agent.modes.review.verdict_folder.ReviewVerdictFolder`.

    Attributes:
        target_kind: Which artefact kind was reviewed.
        overall: The folded outcome — ``pass`` / ``fail`` /
            ``needs_changes``.
        findings: Every :class:`StepFinding` the three checkers produced.
        plan_diff: A proposed repair, present iff ``overall`` is
            ``needs_changes``; ``None`` for ``pass`` / ``fail``.
        intent_ref: The originating ``IntentSpec`` identifier, or ``None``.
        summary: A short human-readable summary of the review.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_kind: ReviewTargetKind
    overall: Outcome
    findings: tuple[StepFinding, ...]
    plan_diff: PlanDiff | None = None
    intent_ref: str | None = None
    summary: str = ""

    def error_findings(self) -> tuple[StepFinding, ...]:
        """Return only the ``error``-severity findings."""
        return tuple(f for f in self.findings if f.severity == "error")


def _derive_outcome(findings: tuple[StepFinding, ...]) -> Outcome:
    """Fold findings into an ``overall`` outcome.

    A finding carrying a ``failed_invariant`` is *actionable* — it can
    drive a repair. Any actionable ``error`` / ``warning`` makes the
    review ``needs_changes`` (a :class:`PlanDiff` is then synthesized).
    An ``error`` with no actionable cause is a hard ``fail`` — there is
    no repair to propose. An all-``info`` (or empty) finding set passes.
    """
    actionable = any(f.failed_invariant and f.severity in ("error", "warning") for f in findings)
    if actionable:
        return "needs_changes"
    if any(f.severity == "error" for f in findings):
        return "fail"
    return "pass"


def _actionable_findings(findings: tuple[StepFinding, ...]) -> tuple[StepFinding, ...]:
    """Return findings that name a ``failed_invariant`` — the repairable ones."""
    return tuple(f for f in findings if f.failed_invariant)


def _build_plan_diff(
    actionable: tuple[StepFinding, ...],
    plan: PlanGraph,
) -> PlanDiff | None:
    """Synthesize a :class:`PlanDiff` from the actionable findings.

    The diff names the worst finding's ``failed_invariant``, the union of
    the findings' ``step_id``s as ``affected_nodes``, and one ``replace``
    op per affected step that still exists in the plan (its planning
    surface must be revised). Surviving steps are ``reused``; the
    transitive dependents of every affected step are ``invalidated``.
    Returns ``None`` when no finding is actionable.
    """
    if not actionable:
        return None
    affected = tuple(dict.fromkeys(f.step_id for f in actionable if f.step_id))
    operations = tuple(
        PlanNodeOp(kind=DiffOpKind.replace, node_id=step_id, step=step)
        for step_id in affected
        if (step := plan.step_by_id(step_id)) is not None
    )
    invalidated: list[str] = []
    for step_id in affected:
        for dependent in plan.downstream_of(step_id):
            if dependent not in invalidated:
                invalidated.append(dependent)
    rationale = "; ".join(f.summary for f in actionable)
    return PlanDiff(
        failed_invariant=actionable[0].failed_invariant or "review_conformance",
        affected_nodes=affected,
        operations=operations,
        rationale=rationale or "the review found a conformance gap",
        reused=tuple(s.id for s in plan.steps if s.id not in affected),
        invalidated=tuple(invalidated),
    )


def build_review_verdict(
    *,
    findings: tuple[StepFinding, ...],
    intent: IntentSpec,
    plan: PlanGraph,
    target_kind: ReviewTargetKind,
    summary: str = "",
) -> ReviewVerdict:
    """Fold review findings into a typed :class:`ReviewVerdict`.

    The ``overall`` outcome comes from :func:`_derive_outcome`. When the
    outcome is ``needs_changes`` a :class:`PlanDiff` is synthesized from
    the actionable findings so the verdict feeds the shared repair loop;
    a ``pass`` / ``fail`` outcome carries no diff (a ``fail`` is a hard
    stop, a ``pass`` needs nothing).

    Args:
        findings: Every :class:`StepFinding` the checkers produced.
        intent: The :class:`IntentSpec` the plan was judged against.
        plan: The :class:`PlanGraph` under review.
        target_kind: The reviewed artefact's :class:`ReviewTargetKind`.
        summary: An optional human-readable summary; a deterministic one
            is derived when empty.

    Returns:
        A frozen :class:`ReviewVerdict`.
    """
    overall = _derive_outcome(findings)
    plan_diff = (
        _build_plan_diff(_actionable_findings(findings), plan)
        if overall == "needs_changes"
        else None
    )
    resolved_summary = summary or _default_summary(overall, findings)
    return ReviewVerdict(
        target_kind=target_kind,
        overall=overall,
        findings=findings,
        plan_diff=plan_diff,
        intent_ref=plan.intent_ref or _intent_objective_ref(intent),
        summary=resolved_summary,
    )


def _default_summary(overall: Outcome, findings: tuple[StepFinding, ...]) -> str:
    """Render a deterministic verdict summary when no LLM line is supplied."""
    error_count = sum(1 for f in findings if f.severity == "error")
    warning_count = sum(1 for f in findings if f.severity == "warning")
    return (
        f"Review outcome: {overall} "
        f"({error_count} error, {warning_count} warning, {len(findings)} finding(s))."
    )


def _intent_objective_ref(intent: IntentSpec) -> str:
    """Fall back to the intent objective as the verdict's intent reference."""
    return intent.objective
