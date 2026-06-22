"""``BindMolcraftsTasks`` — fifth stage of the §3 pipeline.

Asks the configured :class:`AgentGateway` to bind a :class:`WorkflowIR` to
a concrete :class:`BoundWorkflow` (capability assignments + parameters +
execution backend + resource policy). Mirror of Phase-7's
:class:`ExtractWorkflowIR` pattern: ctx-driven gateway, fail-fast on missing.

When the run carries a grounded :class:`CapabilityRegistry`
(``ctx.capability_registry``), the stage renders its catalog into a
``capability_catalog`` artifact and threads it through the gateway call's
``prompt_artifact_id`` — so the binder picks ``capability_id``\\ s that exist in
the toolchain instead of inventing them, keeping its output inside what
``ValidateBoundWorkflow`` will accept. With no registry the call is unchanged.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.prompts.capability_catalog import render_capability_catalog
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, BoundWorkflow
from molexp.harness.stages._resolve import require_latest

__all__ = ["BindMolcraftsTasks"]


class BindMolcraftsTasks(Stage):
    """Bind a WorkflowIR to a BoundWorkflow via gateway."""

    name: ClassVar[str] = "bind_molcrafts_tasks"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("BindMolcraftsTasks requires ctx.agent_gateway to be set")
        ir = require_latest(ctx, "workflow_ir", stage=self.name)

        prompt_artifact_id: str | None = None
        if ctx.capability_registry is not None:
            catalog = render_capability_catalog(ctx.capability_registry.list_capabilities())
            catalog_ref = ctx.artifact_store.put_text(
                kind="capability_catalog",
                text=catalog,
                created_by=f"stage:{self.name}",
                parent_ids=[ir.id],
            )
            prompt_artifact_id = catalog_ref.id

        spec = AgentCallSpec(
            agent_name="bound_workflow_binder",
            input_artifact_ids=[ir.id],
            prompt_artifact_id=prompt_artifact_id,
            output_schema=BoundWorkflow.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
