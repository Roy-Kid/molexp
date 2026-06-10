"""``GenerateTestCode`` — ask the gateway to emit pytest source for the workflow.

Same ctx-driven gateway + fail-fast pattern as :class:`GenerateWorkflowSource`:
builds an :class:`AgentCallSpec` for the ``test_code_writer`` agent with
``output_schema = TestSource.model_json_schema()`` over the TestSpec +
WorkflowSource artifacts, and returns the gateway's persisted
``output_artifact``.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, TestSource
from molexp.harness.stages._resolve import require_latest

__all__ = ["GenerateTestCode"]


class GenerateTestCode(Stage):
    """Generate pytest source realizing the TestSpec via gateway."""

    name: ClassVar[str] = "generate_test_code"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateTestCode requires ctx.agent_gateway to be set")
        test_spec = require_latest(ctx, "test_spec", stage=self.name)
        workflow_source = require_latest(ctx, "workflow_source", stage=self.name)
        spec = AgentCallSpec(
            agent_name="test_code_writer",
            input_artifact_ids=[test_spec.id, workflow_source.id],
            output_schema=TestSource.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
