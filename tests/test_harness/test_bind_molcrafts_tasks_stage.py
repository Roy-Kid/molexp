"""Tests for BindMolcraftsTasks stage (Phase 8). Mirrors ExtractWorkflowIR."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest


def _seed_workflow_ir_ref(artifact_store):
    return artifact_store.put_json(
        kind="workflow_ir",
        obj={
            "id": "wf-x",
            "name": "wf",
            "objective": "x",
            "inputs": {},
            "tasks": [],
            "edges": [],
            "expected_outputs": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _bound_workflow_canned() -> dict:
    return {
        "id": "bw-x",
        "workflow_ir_id": "wf-x",
        "tasks": [],
        "edges": [],
        "execution_backend": "local",
        "environment": {},
        "resource_policy": {
            "backend": "local",
            "max_runtime_s": 3600,
            "denied_paths": ["/", "~/.ssh"],
        },
    }


@pytest.fixture()
def ctx_no_gw(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-bind",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
    )


@pytest.fixture()
def ctx_with_gw(tmp_path: Path):
    from molexp.harness.agents.stub import StubAgentGateway
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    stub = StubAgentGateway(artifact_store=a)
    return HarnessRunContext(
        run_id="run-bind",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
        agent_gateway=stub,
    )


def test_bind_molcrafts_tasks_name() -> None:
    from molexp.harness.stages.bind_molcrafts_tasks import BindMolcraftsTasks

    assert BindMolcraftsTasks.name == "bind_molcrafts_tasks"


def test_bind_molcrafts_tasks_is_stage() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.bind_molcrafts_tasks import BindMolcraftsTasks

    assert issubclass(BindMolcraftsTasks, Stage)


def test_bind_fail_fast_when_gateway_missing(ctx_no_gw) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.bind_molcrafts_tasks import BindMolcraftsTasks

    ir_ref = _seed_workflow_ir_ref(ctx_no_gw.artifact_store)
    stage = BindMolcraftsTasks(workflow_ir_artifact_id=ir_ref.id)
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx_no_gw))
    assert "agent_gateway" in str(exc.value)


def test_bind_builds_correct_spec(ctx_with_gw) -> None:
    from molexp.harness.agents.gateway import AgentGateway
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec, BoundWorkflow
    from molexp.harness.stages.bind_molcrafts_tasks import BindMolcraftsTasks

    ir_ref = _seed_workflow_ir_ref(ctx_with_gw.artifact_store)
    real_gw = ctx_with_gw.agent_gateway
    real_gw.register(
        agent_name="bound_workflow_binder",
        output=_bound_workflow_canned(),
        output_kind="bound_workflow",
    )
    captured: list[AgentCallSpec] = []

    class Capturing:
        async def call(self, spec: AgentCallSpec) -> AgentCallResult:
            captured.append(spec)
            return await real_gw.call(spec)

    object.__setattr__(ctx_with_gw, "_frozen", False)
    ctx_with_gw.agent_gateway = cast(AgentGateway, Capturing())
    object.__setattr__(ctx_with_gw, "_frozen", True)

    stage = BindMolcraftsTasks(workflow_ir_artifact_id=ir_ref.id)
    asyncio.run(stage.run(ctx_with_gw))

    assert len(captured) == 1
    spec = captured[0]
    assert spec.agent_name == "bound_workflow_binder"
    assert spec.input_artifact_ids == [ir_ref.id]
    assert spec.output_schema == BoundWorkflow.model_json_schema()


def test_bind_returns_bound_workflow_ref(ctx_with_gw) -> None:
    from molexp.harness.stages.bind_molcrafts_tasks import BindMolcraftsTasks

    ir_ref = _seed_workflow_ir_ref(ctx_with_gw.artifact_store)
    ctx_with_gw.agent_gateway.register(
        agent_name="bound_workflow_binder",
        output=_bound_workflow_canned(),
        output_kind="bound_workflow",
    )
    stage = BindMolcraftsTasks(workflow_ir_artifact_id=ir_ref.id)
    ref = asyncio.run(stage.run(ctx_with_gw))
    assert ref.kind == "bound_workflow"
    assert ir_ref.id in ref.parent_ids
