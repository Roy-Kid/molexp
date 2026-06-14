"""``FinalReport`` — post-execution experiment report from real results.

The product of :class:`~molexp.harness.stages.generate_final_report.GenerateFinalReport`:
where :class:`~molexp.harness.schemas.experiment_report.ExperimentReport` is
the *plan-time* expansion of the user's goal, ``FinalReport`` is written
*after* the generated tests ran and the workflow actually executed — its
inputs are the experiment report, the ``TestResult``, and the
``ExecutionResult``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["FinalReport"]


class FinalReport(BaseModel):
    """Final experiment report grounded in real test + execution artifacts.

    Attributes:
        title: Report title.
        objective: What the experiment set out to establish.
        methods_summary: How the workflow realized the plan.
        test_summary: What the generated tests covered and their outcome.
        execution_summary: How the execution went (status, runtime, outputs).
        results: The observed results, citing real output values.
        conclusions: What the results support concluding.
        limitations: Known caveats and threats to validity.
        next_steps: Suggested follow-up experiments or refinements.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    objective: str
    methods_summary: str
    test_summary: str
    execution_summary: str
    results: str
    conclusions: str
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
