"""``ExtractWorkflowIR`` — plan step 4: concrete spec → workflow IR.

Asks the gateway on ``ctx.agent_gateway`` to lift the concrete
``ExperimentSpec`` (the parameter-resolved spec from step 2) into a
structured :class:`WorkflowIR` — the flow + topology. Every LLM-backed
stage reads the gateway from ctx — context injection is the one pattern; no
stage takes a gateway constructor arg.

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
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["ExtractWorkflowIR"]


class ExtractWorkflowIR(Stage):
    """Expand a concrete experiment_spec into a structured WorkflowIR via gateway."""

    name: ClassVar[str] = "extract_workflow_ir"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("ExtractWorkflowIR requires ctx.agent_gateway to be set")
        spec_ref = require_latest(ctx, "experiment_spec", stage=self.name)
        call = AgentCallSpec(
            agent_name="workflow_ir_extractor",
            input_artifact_ids=[spec_ref.id, *feedback_inputs(ctx, "workflow_ir_feedback")],
            output_schema=WorkflowIR.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(call)
        return result.output_artifact
