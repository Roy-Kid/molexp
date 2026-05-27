"""End-to-end Phase-2 pipeline test (ac-007).

Runs `SaveUserPlan → GenerateExperimentReport` through a single
StageRunner with a StubAgentGateway and verifies the three-layer
provenance chain `raw_text → user_plan → experiment_report`. This is the
"the harness can actually run a pipeline" deliverable for Phase 2.

Also verifies the public-surface discipline from ac-008:
- Phase-2 symbols importable from molexp.harness top level
- StubAgentGateway NOT importable from molexp.harness or molexp.harness.agents
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness import (
        FileArtifactStore,
        HarnessRunContext,
        SQLiteEventLog,
        SQLiteProvenanceStore,
    )

    db_path = tmp_path / "events.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteProvenanceStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-e2e",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        provenance_store=provenance,
    )


def test_e2e_pipeline_three_layer_provenance(ctx) -> None:
    """SaveUserPlan → GenerateExperimentReport produces a 3-layer chain."""
    from molexp.harness import (
        GenerateExperimentReport,
        SaveUserPlan,
        StageRunner,
    )
    from molexp.harness.agents.stub import StubAgentGateway

    stub = StubAgentGateway(artifact_store=ctx.artifact_store)
    stub.register(
        agent_name="experiment_report_writer",
        output={
            "title": "Water NEMD",
            "objective": "Measure ionic mobility",
            "system_description": "SPC/E water + 64 NaCl pairs",
            "experimental_design": "3 replicas at 300K",
        },
        raw_text="<verbatim LLM response>",
    )

    runner = StageRunner(ctx)
    user_plan_ref = asyncio.run(runner.run_stage(SaveUserPlan(user_text="Simulate water at 300K")))
    report_ref = asyncio.run(
        runner.run_stage(
            GenerateExperimentReport(
                user_plan_artifact_id=user_plan_ref.id,
                gateway=stub,
            )
        )
    )

    # Three-layer chain: raw_text → user_plan → experiment_report.
    ancestors = ctx.provenance_store.trace_backward(report_ref.id)
    assert len(ancestors) == 2
    ids_in_order = [r.id for r in ancestors]
    # BFS: first the immediate parent (user_plan_ref), then the raw text.
    assert ids_in_order[0] == user_plan_ref.id
    raw_text_id = ids_in_order[1]
    raw_ref = ctx.artifact_store.get_ref(raw_text_id)
    assert raw_ref.kind == "user_plan"  # both raw and structured share kind
    # The raw artifact has no parents (it's the audit anchor).
    assert raw_ref.parent_ids == []


def test_e2e_event_log_contains_two_stage_quartets(ctx) -> None:
    from molexp.harness import (
        GenerateExperimentReport,
        SaveUserPlan,
        StageRunner,
    )
    from molexp.harness.agents.stub import StubAgentGateway

    stub = StubAgentGateway(artifact_store=ctx.artifact_store)
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
    user_plan_ref = asyncio.run(runner.run_stage(SaveUserPlan(user_text="hi")))
    asyncio.run(
        runner.run_stage(
            GenerateExperimentReport(
                user_plan_artifact_id=user_plan_ref.id,
                gateway=stub,
            )
        )
    )
    events = ctx.event_log.list_events("run-e2e")
    types = [e.type for e in events]
    assert types == [
        "stage_started",
        "artifact_created",
        "stage_completed",
        "stage_started",
        "artifact_created",
        "stage_completed",
    ]
    # First artifact_created is user_plan, second is experiment_report.
    assert events[1].payload["kind"] == "user_plan"
    assert events[4].payload["kind"] == "experiment_report"


# ---------------------------------------------- Public-surface discipline


def test_phase02_public_symbols_importable_from_top_level() -> None:
    """ac-008 (positive direction)."""
    from molexp.harness import (  # noqa: F401
        AgentCallResult,
        AgentCallSpec,
        AgentGateway,
        ExperimentReport,
        GenerateExperimentReport,
        SaveUserPlan,
        UserPlan,
    )


def test_stub_agent_gateway_not_at_top_level() -> None:
    """ac-008 (negative direction): StubAgentGateway stays private."""
    harness = importlib.import_module("molexp.harness")
    assert not hasattr(harness, "StubAgentGateway")
    agents_pkg = importlib.import_module("molexp.harness.agents")
    assert not hasattr(agents_pkg, "StubAgentGateway")
    # Only reachable via the stub module's full dotted path.
    stub_mod = importlib.import_module("molexp.harness.agents.stub")
    assert hasattr(stub_mod, "StubAgentGateway")
