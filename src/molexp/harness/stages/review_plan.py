"""``ReviewPlan`` — semantic gate: does the workflow implement the report?

Runs after ``ValidateWorkflowSource``. Where the structural validators prove the
plan is a well-formed, compilable DAG, this stage asks the ``plan_reviewer``
agent to compare the experiment report (requirements) against the generated
workflow source (implementation) under a fixed domain-agnostic rubric, and
fails the plan when the workflow drops, zeroes, stubs, or contradicts a stated
requirement — the gap that let a ``charge=0.0`` zwitterion pass every structural
check.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, PlanReview
from molexp.harness.stages._resolve import require_latest

__all__ = ["ReviewPlan"]

_MAX_REPORTED = 8


class ReviewPlan(Stage):
    """LLM judge: fail the plan when the workflow is unfaithful to the report."""

    name: ClassVar[str] = "review_plan"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("ReviewPlan requires ctx.agent_gateway to be set")
        report = require_latest(ctx, "experiment_report", stage=self.name)
        source = require_latest(ctx, "workflow_source", stage=self.name)
        spec = AgentCallSpec(
            agent_name="plan_reviewer",
            input_artifact_ids=[report.id, source.id],
            output_schema=PlanReview.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)

        review = PlanReview.model_validate_json(ctx.artifact_store.get(result.output_artifact.id))
        errors = [f for f in review.findings if f.severity == "error"]
        # The stage — not the model's self-reported ``passed`` — is the gate: any
        # error-severity finding fails the plan even if the model set passed=True.
        if errors or not review.passed:
            deviations = "; ".join(
                f"{f.requirement} -> {f.deviation}" for f in errors[:_MAX_REPORTED]
            )
            # Persisted-failure so RepairLoop reads the review (``persisted_ref``)
            # as feedback for the next workflow-source attempt.
            raise StagePersistedFailureError(
                result.output_artifact,
                "plan review failed — the workflow does not faithfully implement the "
                f"experiment report ({len(errors)} error finding(s)): {deviations}",
            )
        return result.output_artifact
