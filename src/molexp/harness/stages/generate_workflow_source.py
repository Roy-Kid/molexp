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

__all__ = ["GenerateWorkflowSource"]


class GenerateWorkflowSource(Stage):
    """Generate runnable molexp.workflow source for a BoundWorkflow via gateway."""

    name: ClassVar[str] = "generate_workflow_source"

    def __init__(self, bound_workflow_artifact_id: str) -> None:
        self._bound_workflow_artifact_id = bound_workflow_artifact_id

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateWorkflowSource requires ctx.agent_gateway to be set")
        spec = AgentCallSpec(
            agent_name="workflow_source_writer",
            input_artifact_ids=[self._bound_workflow_artifact_id],
            output_schema=WorkflowSource.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
