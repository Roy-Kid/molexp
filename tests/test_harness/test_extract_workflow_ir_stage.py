"""Tests for ExtractWorkflowIR stage (Phase 7).

Locks:
- Stage subclass, name="extract_workflow_ir"
- resolves latest experiment_report from the store (no ctor args)
- fail-fast on ctx.agent_gateway is None
- builds AgentCallSpec(agent_name="workflow_ir_extractor", ...) with correct input_artifact_ids + output_schema
- returns gateway.output_artifact (kind=workflow_ir, parent_ids=[experiment_report_artifact_id])
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest


def _seed_experiment_report_ref(artifact_store):
    """Seed an experiment_report artifact the stage will reference."""
    return artifact_store.put_json(
        kind="experiment_report",
        obj={
            "title": "t",
            "objective": "o",
            "system_description": "s",
            "experimental_design": "e",
        },
        created_by="seed",
        parent_ids=[],
    )


@pytest.fixture()
def ctx_no_gateway(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-extract",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
    )


@pytest.fixture()
def ctx_with_gateway(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    stub = StubAgentGateway(artifact_store=a)
    return HarnessRunContext(
        run_id="run-extract",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
        agent_gateway=stub,
    )


def test_extract_workflow_ir_name() -> None:
    from molexp.harness.stages.extract_workflow_ir import ExtractWorkflowIR

    assert ExtractWorkflowIR.name == "extract_workflow_ir"


def test_extract_workflow_ir_is_stage_subclass() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.extract_workflow_ir import ExtractWorkflowIR

    assert issubclass(ExtractWorkflowIR, Stage)


def test_extract_fail_fast_when_gateway_missing(ctx_no_gateway) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.extract_workflow_ir import ExtractWorkflowIR

    _seed_experiment_report_ref(ctx_no_gateway.artifact_store)
    stage = ExtractWorkflowIR()
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx_no_gateway))
    assert "agent_gateway" in str(exc.value)


def _make_workflow_ir_canned_response() -> dict:
    return {
        "id": "wf-x",
        "name": "wf",
        "objective": "x",
        "inputs": {},
        "tasks": [
            {
                "id": "t1",
                "name": "T",
                "purpose": "p",
                "task_type": "tt",
                "inputs": {},
                "outputs": {"out": "out.txt"},
            }
        ],
        "edges": [],
        "expected_outputs": [],
    }


def test_extract_builds_correct_spec(ctx_with_gateway) -> None:
    """Capture the spec the stage passes to the gateway."""
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec, WorkflowIR
    from molexp.harness.stages.extract_workflow_ir import ExtractWorkflowIR

    er_ref = _seed_experiment_report_ref(ctx_with_gateway.artifact_store)
    real_gateway = ctx_with_gateway.agent_gateway
    real_gateway.register(
        agent_name="workflow_ir_extractor",
        output=_make_workflow_ir_canned_response(),
        output_kind="workflow_ir",
    )
    captured: list[AgentCallSpec] = []

    class CapturingGateway:
        async def call(self, spec: AgentCallSpec) -> AgentCallResult:
            captured.append(spec)
            return await real_gateway.call(spec)

    object.__setattr__(ctx_with_gateway, "_frozen", False)
    ctx_with_gateway.agent_gateway = cast(AgentGateway, CapturingGateway())
    object.__setattr__(ctx_with_gateway, "_frozen", True)

    stage = ExtractWorkflowIR()
    asyncio.run(stage.run(ctx_with_gateway))
    assert len(captured) == 1
    spec = captured[0]
    assert spec.agent_name == "workflow_ir_extractor"
    assert spec.input_artifact_ids == [er_ref.id]
    assert spec.output_schema == WorkflowIR.model_json_schema()


def test_extract_returns_workflow_ir_ref_with_parent_ids(ctx_with_gateway) -> None:
    from molexp.harness.stages.extract_workflow_ir import ExtractWorkflowIR

    er_ref = _seed_experiment_report_ref(ctx_with_gateway.artifact_store)
    ctx_with_gateway.agent_gateway.register(
        agent_name="workflow_ir_extractor",
        output=_make_workflow_ir_canned_response(),
        output_kind="workflow_ir",
    )
    stage = ExtractWorkflowIR()
    result_ref = asyncio.run(stage.run(ctx_with_gateway))
    assert result_ref.kind == "workflow_ir"
    assert er_ref.id in result_ref.parent_ids
