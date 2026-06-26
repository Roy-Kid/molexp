"""``PlanReview`` — an LLM judge's verdict on whether a generated workflow
faithfully implements its experiment report.

The structural validators (``ValidateWorkflowIR`` / ``ValidateBoundWorkflow`` /
``ValidateWorkflowSource``) only prove the plan is a well-formed, compilable DAG
— they cannot tell that a *zwitterionic* system was built with ``charge=0.0`` or
that a required packing step is missing. ``ReviewPlan`` closes that semantic gap
by asking an LLM to compare the experiment report (the requirements) against the
generated workflow (the implementation) under a fixed, domain-agnostic rubric;
this is its structured output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PlanReview", "PlanReviewFinding"]


class PlanReviewFinding(BaseModel):
    """One way the workflow diverges from the experiment report."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning"]
    requirement: str
    """The report requirement this finding is about (quoted/paraphrased)."""
    deviation: str
    """How the workflow drops, weakens, stubs, or contradicts that requirement."""


class PlanReview(BaseModel):
    """The reviewer's verdict on report-vs-workflow fidelity.

    ``passed`` is the model's own call; the ``ReviewPlan`` stage independently
    fails the plan when any ``error``-severity finding is present, so a model
    that lists errors but sets ``passed=True`` cannot slip a broken plan through.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    findings: list[PlanReviewFinding] = Field(default_factory=list)
    summary: str = ""
