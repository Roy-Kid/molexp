"""``GenerateTestSpec`` — seventh stage of the §3 pipeline.

Asks the configured :class:`AgentGateway` to author a :class:`TestSpec`
for the supplied BoundWorkflow. Same ctx-driven gateway + fail-fast
pattern as Phase-7's :class:`ExtractWorkflowIR` and Phase-8's
:class:`BindMolcraftsTasks`.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, TestSpec

__all__ = ["GenerateTestSpec"]


class GenerateTestSpec(Stage):
    """Generate a TestSpec for a BoundWorkflow via gateway."""

    name: ClassVar[str] = "generate_test_spec"

    def __init__(self, bound_workflow_artifact_id: str) -> None:
        self._bound_workflow_artifact_id = bound_workflow_artifact_id

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateTestSpec requires ctx.agent_gateway to be set")
        spec = AgentCallSpec(
            agent_name="test_spec_writer",
            input_artifact_ids=[self._bound_workflow_artifact_id],
            output_schema=TestSpec.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
