"""``GenerateWorkflowSource`` — ask the gateway to emit molexp.workflow source.

Same ctx-driven gateway + fail-fast pattern as :class:`GenerateTestSpec`:
builds an :class:`AgentCallSpec` for the ``workflow_source_writer`` agent with
``output_schema = WorkflowSource.model_json_schema()`` over the BoundWorkflow
artifact, and returns the gateway's persisted ``output_artifact``.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, WorkflowSource
from molexp.harness.stages._resolve import require_latest

__all__ = ["GenerateWorkflowSource"]


class GenerateWorkflowSource(Stage):
    """Generate runnable molexp.workflow source for a BoundWorkflow via gateway."""

    name: ClassVar[str] = "generate_workflow_source"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateWorkflowSource requires ctx.agent_gateway to be set")
        bound = require_latest(ctx, "bound_workflow", stage=self.name)
        spec = AgentCallSpec(
            agent_name="workflow_source_writer",
            input_artifact_ids=[bound.id],
            output_schema=WorkflowSource.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
