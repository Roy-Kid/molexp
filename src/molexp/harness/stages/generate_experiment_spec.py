"""``GenerateExperimentSpec`` — plan step 2: report → concrete spec.

Asks the gateway to lift the human-readable :class:`ExperimentReport` into a
concrete :class:`ExperimentSpec`: every free-text variable / condition
becomes a provenance-carrying :class:`ParameterValue`, and every open
``user_questions`` entry is given a resolved answer. Same ctx-driven
gateway + fail-fast pattern as :class:`ExtractWorkflowIR`.

The stored artifact is JSON (kind ``experiment_spec``); the human-readable
YAML view is rendered from it at the CLI/server boundary, not duplicated
into the store (it is a derived view, not a second source of truth).
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, ExperimentSpec
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["GenerateExperimentSpec"]


class GenerateExperimentSpec(Stage):
    """Expand an experiment_report into a concrete ExperimentSpec via gateway."""

    name: ClassVar[str] = "generate_experiment_spec"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateExperimentSpec requires ctx.agent_gateway to be set")
        report = require_latest(ctx, "experiment_report", stage=self.name)
        spec = AgentCallSpec(
            agent_name="experiment_spec_generator",
            input_artifact_ids=[report.id, *feedback_inputs(ctx, "experiment_spec_feedback")],
            output_schema=ExperimentSpec.model_json_schema(),
        )
        result = await ctx.agent_gateway.call(spec)
        return result.output_artifact
