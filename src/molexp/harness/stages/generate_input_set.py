"""``GenerateInputSet`` — plan step 6: spec → parameter-space expansion.

Asks the gateway to turn the concrete :class:`ExperimentSpec` (and the
:class:`WorkflowIR` it produced) into an :class:`InputSet`: which root
inputs are swept and over what values. The harness only *describes* the
sweep; the workspace ``ParamSpace`` family expands it. Same ctx-driven
gateway + fail-fast pattern as the other LLM stages.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, InputSet
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["GenerateInputSet"]


class GenerateInputSet(Stage):
    """Expand an experiment_spec into a sweep InputSet via gateway."""

    name: ClassVar[str] = "generate_input_set"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateInputSet requires ctx.agent_gateway to be set")
        spec = require_latest(ctx, "experiment_spec", stage=self.name)
        ir = require_latest(ctx, "workflow_ir", stage=self.name)
        call = AgentCallSpec(
            agent_name="input_set_generator",
            input_artifact_ids=[spec.id, ir.id, *feedback_inputs(ctx, "input_set_feedback")],
            output_schema=InputSet.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(call)
        return result.output_artifact
