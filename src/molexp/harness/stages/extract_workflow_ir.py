"""``ExtractWorkflowIR`` — third stage of the §3 pipeline.

Asks the configured :class:`AgentGateway` to expand an
``ExperimentReport`` artifact into a structured :class:`WorkflowIR`.

Unlike Phase-2's :class:`GenerateExperimentReport` (which takes the
gateway as a constructor arg because Phase-1 ``HarnessRunContext`` had
no ``agent_gateway`` field), Phase 7 ships the field and reads it from
ctx. The architectural inflection point: ctor-injection (Phase 2) and
ctx-injection (Phase 7+) coexist temporarily — a future cleanup phase
will migrate Phase-2's stage to the ctx-based pattern.

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
