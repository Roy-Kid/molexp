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
from molexp.harness.prompts.capability_catalog import render_capability_catalog
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, WorkflowSource
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["GenerateWorkflowSource"]


class GenerateWorkflowSource(Stage):
    """Generate runnable molexp.workflow source for a BoundWorkflow via gateway.

    When the run carries a grounded :class:`CapabilityRegistry`, the catalog (with
    each capability's call signature) is threaded into the writer's prompt so the
    generated task bodies invoke the bound molpy capabilities with real arguments
    instead of emitting stubs.
    """

    name: ClassVar[str] = "generate_workflow_source"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateWorkflowSource requires ctx.agent_gateway to be set")
        bound = require_latest(ctx, "bound_workflow", stage=self.name)

        prompt_artifact_id: str | None = None
        if ctx.capability_registry is not None:
            catalog = render_capability_catalog(ctx.capability_registry.list_capabilities())
            catalog_ref = ctx.artifact_store.put_text(
                kind="capability_catalog",
                text=catalog,
                created_by=f"stage:{self.name}",
                parent_ids=[bound.id],
            )
            prompt_artifact_id = catalog_ref.id

        spec = AgentCallSpec(
            agent_name="workflow_source_writer",
            input_artifact_ids=[bound.id, *feedback_inputs(ctx, "workflow_source_feedback")],
            prompt_artifact_id=prompt_artifact_id,
            output_schema=WorkflowSource.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
