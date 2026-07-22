"""Tests for the ``GenerateExperimentReport`` pipeline stage.

Owned behaviors (harness pipeline orchestration):
- builds an :class:`AgentCallSpec` for ``experiment_report_writer`` whose
  ordered inputs are ``[user_plan, knowledge_context]`` and whose
  ``output_schema`` is ``ExperimentReport.model_json_schema()``;
- returns the gateway's ``output_artifact`` (kind ``experiment_report``) and,
  through the :class:`StageRunner`, wires the ``user_plan → experiment_report``
  ``derived_from`` lineage edge.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db_path = tmp_path / "events.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    gateway = StubAgentGateway(artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-gen-report",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=provenance,
        agent_gateway=gateway,
    )


@pytest.fixture()
def user_plan_ref(ctx):
    """Seed the knowledge_context + user_plan artifacts the stage consumes."""
    # The pipeline's AssembleKnowledgeContext always precedes this stage —
    # standalone stage tests seed the knowledge_context artifact it provides.
    ctx.artifact_store.put_text(
        kind="knowledge_context",
        text="no prior knowledge recorded in this workspace",
        created_by="seed",
        parent_ids=[],
    )
    return ctx.artifact_store.put_json(
        kind="user_plan",
        obj={"raw_text": "simulate water", "submitted_at": "2026-05-26T00:00:00Z"},
        created_by="seed",
        parent_ids=[],
    )


@pytest.fixture()
def stub(ctx):
    return ctx.agent_gateway


_CANNED_REPORT = {
    "title": "t",
    "objective": "o",
    "system_description": "s",
    "experimental_design": "e",
}


class TestGenerateExperimentReport:
    def test_builds_spec_with_user_plan_then_knowledge_and_schema(
        self, ctx, user_plan_ref, stub
    ) -> None:
        from molexp.harness.core.stage_runner import StageRunner
        from molexp.harness.gateways.gateway import AgentGateway
        from molexp.harness.schemas import AgentCallResult, AgentCallSpec
        from molexp.harness.schemas.experiment_report import ExperimentReport
        from molexp.harness.stages.generate_experiment_report import GenerateExperimentReport

        captured: list[AgentCallSpec] = []
        real_call = stub.call

        class CapturingGateway:
            async def call(self, spec: AgentCallSpec) -> AgentCallResult:
                captured.append(spec)
                return await real_call(spec)

        stub.register(agent_name="experiment_report_writer", output=_CANNED_REPORT)

        object.__setattr__(ctx, "_frozen", False)
        ctx.agent_gateway = cast(AgentGateway, CapturingGateway())
        object.__setattr__(ctx, "_frozen", True)
        runner = StageRunner(ctx)
        asyncio.run(runner.run_stage(GenerateExperimentReport()))

        assert len(captured) == 1
        spec = captured[0]
        assert spec.agent_name == "experiment_report_writer"
        # user_plan first, then the knowledge_context digest (vision-loop-05).
        assert spec.input_artifact_ids[0] == user_plan_ref.id
        assert len(spec.input_artifact_ids) == 2
        assert spec.output_schema == ExperimentReport.model_json_schema()

    def test_wires_user_plan_provenance(self, ctx, user_plan_ref, stub) -> None:
        from molexp.harness.core.stage_runner import StageRunner
        from molexp.harness.stages.generate_experiment_report import GenerateExperimentReport

        stub.register(agent_name="experiment_report_writer", output=_CANNED_REPORT)
        runner = StageRunner(ctx)
        report_ref = asyncio.run(runner.run_stage(GenerateExperimentReport()))

        assert report_ref.kind == "experiment_report"
        assert user_plan_ref.id in report_ref.parent_ids
        ancestors = ctx.lineage_store.trace_backward(report_ref.id)
        assert user_plan_ref.id in {r.id for r in ancestors}
