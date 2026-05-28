"""``GenerateExperimentReport`` — second stage of the §3 pipeline.

Builds an :class:`AgentCallSpec` that asks the configured
:class:`AgentGateway` for a structured :class:`ExperimentReport` derived
from the given ``user_plan`` artifact, then returns the gateway's parsed
``output_artifact`` unchanged. The gateway is responsible for persisting
both the parsed output and the raw response and for wiring
``parent_ids`` so :class:`StageRunner` materializes the
``user_plan → experiment_report`` ``derived_from`` edge automatically.

Phase 2 wires the gateway via constructor injection (not through
:class:`HarnessRunContext`) because the real LLM-backed gateway impl is
still Phase 5+; once it lands, the gateway moves into the context and this
stage's constructor loses the ``gateway`` arg.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.gateways.gateway import AgentGateway
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, ExperimentReport

__all__ = ["GenerateExperimentReport"]


class GenerateExperimentReport(Stage):
    """Ask the gateway to expand a user plan into a structured report."""

    name: ClassVar[str] = "generate_experiment_report"

    def __init__(
        self,
        user_plan_artifact_id: str,
        gateway: AgentGateway,
    ) -> None:
        self._user_plan_artifact_id = user_plan_artifact_id
        self._gateway = gateway

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:  # noqa: ARG002  — `ctx` is the Stage ABC's contract; this stage delegates to its injected gateway, which already holds an ArtifactStore reference
        spec = AgentCallSpec(
            agent_name="experiment_report_writer",
            input_artifact_ids=[self._user_plan_artifact_id],
            output_schema=ExperimentReport.model_json_schema(),
        )
        result = await self._gateway.call(spec)
        return result.output_artifact
