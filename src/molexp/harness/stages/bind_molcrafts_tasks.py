"""``BindMolcraftsTasks`` — plan step 5: bind IR tasks to capabilities.

Asks the configured :class:`AgentGateway` to bind a :class:`WorkflowIR` to
a concrete :class:`BoundWorkflow` (capability assignments + parameters +
execution backend + resource policy). Mirror of :class:`ExtractWorkflowIR`'s
pattern: ctx-driven gateway, fail-fast on missing.

The ``capability_catalog`` is produced upstream by the dedicated
:class:`ResolveCapabilities` step (plan step 3); this stage consumes the
latest one and threads it through the gateway call's ``prompt_artifact_id``
— so the binder picks ``capability_id``\\ s that exist in the toolchain
instead of inventing them, keeping its output inside what
``ValidateBoundWorkflow`` will accept. If no catalog exists (capability
resolution skipped), the call proceeds unguided.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, BoundWorkflow
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["BindMolcraftsTasks"]


class BindMolcraftsTasks(Stage):
    """Bind a WorkflowIR to a BoundWorkflow via gateway."""

    name: ClassVar[str] = "bind_molcrafts_tasks"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("BindMolcraftsTasks requires ctx.agent_gateway to be set")
        ir = require_latest(ctx, "workflow_ir", stage=self.name)

        catalog_ref = ctx.artifact_store.latest_by_kind("capability_catalog")
        prompt_artifact_id = catalog_ref.id if catalog_ref is not None else None

        spec = AgentCallSpec(
            agent_name="bound_workflow_binder",
            input_artifact_ids=[ir.id, *feedback_inputs(ctx, "bound_workflow_feedback")],
            prompt_artifact_id=prompt_artifact_id,
            output_schema=BoundWorkflow.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
