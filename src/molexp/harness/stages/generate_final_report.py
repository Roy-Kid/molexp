"""``GenerateFinalReport`` — write the post-execution experiment report.

Same ctx-driven gateway + fail-fast pattern as :class:`GenerateTestCode`:
builds an :class:`AgentCallSpec` for the ``final_report_writer`` agent with
``output_schema = FinalReport.model_json_schema()`` over the plan-time
experiment report plus the *real* test + execution results, and returns the
gateway's persisted ``output_artifact``. This is the report-from-real-results
counterpart of the plan-time :class:`GenerateExperimentReport`.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, FinalReport
from molexp.harness.stages._resolve import require_latest

__all__ = ["GenerateFinalReport"]


class GenerateFinalReport(Stage):
    """Generate a FinalReport from real test + execution artifacts via gateway."""

    name: ClassVar[str] = "generate_final_report"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateFinalReport requires ctx.agent_gateway to be set")
        experiment_report = require_latest(ctx, "experiment_report", stage=self.name)
        test_result = require_latest(ctx, "test_result", stage=self.name)
        execution_result = require_latest(ctx, "execution_result", stage=self.name)
        spec = AgentCallSpec(
            agent_name="final_report_writer",
            input_artifact_ids=[experiment_report.id, test_result.id, execution_result.id],
            output_schema=FinalReport.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
