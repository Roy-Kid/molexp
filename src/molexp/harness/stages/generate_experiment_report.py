"""``GenerateExperimentReport`` — second stage of the §3 pipeline.

Builds an :class:`AgentCallSpec` that asks the gateway on
``ctx.agent_gateway`` for a structured :class:`ExperimentReport` derived
from the given ``user_plan`` artifact, then returns the gateway's parsed
``output_artifact`` unchanged. The gateway is responsible for persisting
both the parsed output and the raw response and for wiring ``parent_ids``
so the audit bracket materializes the ``user_plan → experiment_report``
``derived_from`` edge automatically.

Fail-fast: if ``ctx.agent_gateway is None``, the stage raises
:class:`StageExecutionError` rather than NPE-ing midway through the
:class:`AgentCallSpec` construction.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, ExperimentReport
from molexp.harness.stages._resolve import require_latest

__all__ = ["GenerateExperimentReport"]


class GenerateExperimentReport(Stage):
    """Ask the gateway to expand a user plan into a structured report."""

    name: ClassVar[str] = "generate_experiment_report"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError(
                "GenerateExperimentReport requires ctx.agent_gateway to be set"
            )
        user_plan = require_latest(ctx, "user_plan", stage=self.name)
        spec = AgentCallSpec(
            agent_name="experiment_report_writer",
            input_artifact_ids=[user_plan.id],
            output_schema=ExperimentReport.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
