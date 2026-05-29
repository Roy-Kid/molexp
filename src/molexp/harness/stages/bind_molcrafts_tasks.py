"""``BindMolcraftsTasks`` — fifth stage of the §3 pipeline.

Asks the configured :class:`AgentGateway` to bind a :class:`WorkflowIR` to
a concrete :class:`BoundWorkflow` (capability assignments + parameters +
execution backend + resource policy). Mirror of Phase-7's
:class:`ExtractWorkflowIR` pattern: ctx-driven gateway, fail-fast on missing.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
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
        spec = AgentCallSpec(
            agent_name="bound_workflow_binder",
            input_artifact_ids=[ir.id],
            output_schema=BoundWorkflow.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
