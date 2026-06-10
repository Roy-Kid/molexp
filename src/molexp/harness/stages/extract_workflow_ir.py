"""``ExtractWorkflowIR`` — third stage of the §3 pipeline.

Asks the gateway on ``ctx.agent_gateway`` to expand an
``ExperimentReport`` artifact into a structured :class:`WorkflowIR`.
Every LLM-backed stage reads the gateway from ctx — context injection is
the one pattern; no stage takes a gateway constructor arg.

Fail-fast: if ``ctx.agent_gateway is None``, the stage raises
:class:`StageExecutionError` rather than NPE-ing midway through the
:class:`AgentCallSpec` construction.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, WorkflowIR
from molexp.harness.stages._resolve import require_latest

__all__ = ["ExtractWorkflowIR"]


class ExtractWorkflowIR(Stage):
    """Expand an experiment_report into a structured WorkflowIR via gateway."""

    name: ClassVar[str] = "extract_workflow_ir"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("ExtractWorkflowIR requires ctx.agent_gateway to be set")
        report = require_latest(ctx, "experiment_report", stage=self.name)
        spec = AgentCallSpec(
            agent_name="workflow_ir_extractor",
            input_artifact_ids=[report.id],
            output_schema=WorkflowIR.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
