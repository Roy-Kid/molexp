"""Human-readable rendering for a :class:`ReviewVerdict`.

:func:`render_verdict_markdown` turns the typed verdict into the
``verdict.md`` document
:class:`~molexp.agent.modes.review.verdict_folder.ReviewVerdictFolder`
persists alongside the structured ``verdict.yaml``. Pure string
formatting — no LLM, no I/O.
"""

from __future__ import annotations

from molexp.agent.modes.review.verdict import ReviewVerdict, StepFinding

__all__ = ["render_verdict_markdown"]

_SEVERITY_BADGE = {"info": "INFO", "warning": "WARN", "error": "ERROR"}


def render_verdict_markdown(verdict: ReviewVerdict) -> str:
    """Render a :class:`ReviewVerdict` as a Markdown document.

    Args:
        verdict: The verdict to render.

    Returns:
        A Markdown string with a header, the per-step findings, and —
        when present — the proposed :class:`PlanDiff` summary.
    """
    lines: list[str] = [
        f"# Review verdict — {verdict.overall}",
        "",
        f"- Target kind: `{verdict.target_kind.value}`",
        f"- Intent: {verdict.intent_ref or '(none)'}",
        f"- Findings: {len(verdict.findings)}",
        "",
        verdict.summary or "(no summary)",
        "",
        "## Findings",
        "",
    ]
    if not verdict.findings:
        lines.append("_No findings — the artefact conforms._")
    else:
        lines.extend(_render_finding(finding) for finding in verdict.findings)
    if verdict.plan_diff is not None:
        lines.extend(_render_plan_diff(verdict))
    return "\n".join(lines) + "\n"


def _render_finding(finding: StepFinding) -> str:
    """Render one :class:`StepFinding` as a Markdown list item."""
    badge = _SEVERITY_BADGE.get(finding.severity, finding.severity.upper())
    scope = f"`{finding.step_id}`" if finding.step_id else "_plan-wide_"
    detail = f" — {finding.detail}" if finding.detail else ""
    return f"- **[{badge}]** {scope}: {finding.summary}{detail}"


def _render_plan_diff(verdict: ReviewVerdict) -> list[str]:
    """Render the proposed :class:`PlanDiff` as a Markdown section."""
    diff = verdict.plan_diff
    assert diff is not None  # narrowed by the caller
    return [
        "",
        "## Proposed repair (PlanDiff)",
        "",
        f"- Failed invariant: `{diff.failed_invariant}`",
        f"- Affected nodes: {', '.join(diff.affected_nodes) or '(none)'}",
        f"- Operations: {len(diff.operations)}",
        f"- Rationale: {diff.rationale}",
    ]
