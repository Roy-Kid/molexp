"""``GenerateTestSpec`` — seventh stage of the §3 pipeline.

Asks the configured :class:`AgentGateway` to author a
:class:`TestSpecBundle` — **one :class:`TestSpec` per ``BoundTask``** of the
supplied BoundWorkflow — persisted as a single ``test_spec`` artifact (the
bundle keeps the single-latest-artifact contract that ``ValidateTestSpec``
and ``GenerateTestCode`` rely on). Same ctx-driven gateway + fail-fast
pattern as Phase-7's :class:`ExtractWorkflowIR` and Phase-8's
:class:`BindMolcraftsTasks`. On a :class:`RepairLoop` retry the previous
attempt's ``test_spec_feedback`` (the validator's violations) is threaded in
as an extra input, mirroring :class:`GenerateWorkflowSource`.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, PlanArtifactRef, TestSpecBundle
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["GenerateTestSpec"]


class GenerateTestSpec(Stage):
    """Generate a per-task TestSpecBundle for a BoundWorkflow via gateway."""

    name: ClassVar[str] = "generate_test_spec"

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateTestSpec requires ctx.agent_gateway to be set")
        bound = require_latest(ctx, "bound_workflow", stage=self.name)
        spec = AgentCallSpec(
            agent_name="test_spec_writer",
            input_artifact_ids=[bound.id, *feedback_inputs(ctx, "test_spec_feedback")],
            output_schema=TestSpecBundle.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
