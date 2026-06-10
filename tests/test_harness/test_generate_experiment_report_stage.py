"""Tests for GenerateExperimentReport stage (Phase 2 §GenerateExperimentReport).

Locks:
- name == "generate_experiment_report"
- Builds AgentCallSpec(agent_name="experiment_report_writer",
  input_artifact_ids=[user_plan_id], output_schema=ExperimentReport.model_json_schema())
- Returns gateway.call() result.output_artifact unchanged
- Through StageRunner: user_plan → experiment_report derived_from edge wired
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
    """Seed a user_plan artifact the stage will reference as upstream."""
    return ctx.artifact_store.put_json(
        kind="user_plan",
        obj={"raw_text": "simulate water", "submitted_at": "2026-05-26T00:00:00Z"},
        created_by="seed",
        parent_ids=[],
    )


@pytest.fixture()
def stub(ctx):
    return ctx.agent_gateway


def test_generate_experiment_report_name() -> None:
    from molexp.harness.stages.generate_experiment_report import GenerateExperimentReport

    assert GenerateExperimentReport.name == "generate_experiment_report"


def test_generate_experiment_report_builds_correct_spec(ctx, user_plan_ref, stub) -> None:
    """The stage must hand the gateway a spec wired to the user_plan
    artifact and carrying ExperimentReport.model_json_schema()."""
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

    stub.register(
        agent_name="experiment_report_writer",
        output={
            "title": "t",
            "objective": "o",
            "system_description": "s",
            "experimental_design": "e",
        },
    )

    object.__setattr__(ctx, "_frozen", False)
    ctx.agent_gateway = cast(AgentGateway, CapturingGateway())
    object.__setattr__(ctx, "_frozen", True)
    runner = StageRunner(ctx)
    asyncio.run(runner.run_stage(GenerateExperimentReport()))

    assert len(captured) == 1
    spec = captured[0]
    assert spec.agent_name == "experiment_report_writer"
    assert spec.input_artifact_ids == [user_plan_ref.id]
    assert spec.output_schema == ExperimentReport.model_json_schema()


def test_generate_experiment_report_wires_provenance(ctx, user_plan_ref, stub) -> None:
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.stages.generate_experiment_report import GenerateExperimentReport

    stub.register(
        agent_name="experiment_report_writer",
        output={
            "title": "t",
            "objective": "o",
            "system_description": "s",
            "experimental_design": "e",
        },
    )
    runner = StageRunner(ctx)
    report_ref = asyncio.run(runner.run_stage(GenerateExperimentReport()))
    assert report_ref.kind == "experiment_report"
    assert user_plan_ref.id in report_ref.parent_ids
    ancestors = ctx.lineage_store.trace_backward(report_ref.id)
    assert user_plan_ref.id in {r.id for r in ancestors}


def test_generate_experiment_report_event_log_quartet(ctx, user_plan_ref, stub) -> None:
    from molexp.harness.core.stage_runner import StageRunner
    from molexp.harness.stages.generate_experiment_report import GenerateExperimentReport

    stub.register(
        agent_name="experiment_report_writer",
        output={
            "title": "t",
            "objective": "o",
            "system_description": "s",
            "experimental_design": "e",
        },
    )
    runner = StageRunner(ctx)
    asyncio.run(runner.run_stage(GenerateExperimentReport()))
    events = ctx.event_log.list_events("run-gen-report")
    assert [e.type for e in events] == [
        "stage_started",
        "artifact_created",
        "stage_completed",
    ]
    assert events[1].payload["kind"] == "experiment_report"
