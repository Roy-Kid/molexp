"""Tests for GenerateTestSpec stage (Phase 8)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest


def _seed_bw_ref(artifact_store):
    return artifact_store.put_json(
        kind="bound_workflow",
        obj={
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
        },
        created_by="seed",
        parent_ids=[],
    )


def _test_spec_canned() -> dict:
    return {
        "id": "ts-001",
        "name": "dry-run",
        "kind": "dry_run_test",
        "description": "verify all tasks parse",
        "target_workflow_id": "wf-x",
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
        run_id="run-gts",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
    )


@pytest.fixture()
def ctx_with_gw(tmp_path: Path):
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
        run_id="run-gts",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
        agent_gateway=stub,
    )


def test_name_and_subclass() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.generate_test_spec import GenerateTestSpec

    assert GenerateTestSpec.name == "generate_test_spec"
    assert issubclass(GenerateTestSpec, Stage)


def test_fail_fast_no_gateway(ctx_no_gw) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.generate_test_spec import GenerateTestSpec

    bw_ref = _seed_bw_ref(ctx_no_gw.artifact_store)
    stage = GenerateTestSpec(bound_workflow_artifact_id=bw_ref.id)
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx_no_gw))
    assert "agent_gateway" in str(exc.value)


def test_builds_correct_spec(ctx_with_gw) -> None:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec, TestSpec
    from molexp.harness.stages.generate_test_spec import GenerateTestSpec

    bw_ref = _seed_bw_ref(ctx_with_gw.artifact_store)
    real_gw = ctx_with_gw.agent_gateway
    real_gw.register(
        agent_name="test_spec_writer",
        output=_test_spec_canned(),
        output_kind="test_spec",
    )
    captured: list[AgentCallSpec] = []

    class Cap:
        async def call(self, spec: AgentCallSpec) -> AgentCallResult:
            captured.append(spec)
            return await real_gw.call(spec)

    object.__setattr__(ctx_with_gw, "_frozen", False)
    ctx_with_gw.agent_gateway = cast(AgentGateway, Cap())
    object.__setattr__(ctx_with_gw, "_frozen", True)

    asyncio.run(GenerateTestSpec(bound_workflow_artifact_id=bw_ref.id).run(ctx_with_gw))
    assert len(captured) == 1
    spec = captured[0]
    assert spec.agent_name == "test_spec_writer"
    assert spec.input_artifact_ids == [bw_ref.id]
    assert spec.output_schema == TestSpec.model_json_schema()


def test_returns_test_spec_ref(ctx_with_gw) -> None:
    from molexp.harness.stages.generate_test_spec import GenerateTestSpec

    bw_ref = _seed_bw_ref(ctx_with_gw.artifact_store)
    ctx_with_gw.agent_gateway.register(
        agent_name="test_spec_writer",
        output=_test_spec_canned(),
        output_kind="test_spec",
    )
    ref = asyncio.run(GenerateTestSpec(bound_workflow_artifact_id=bw_ref.id).run(ctx_with_gw))
    assert ref.kind == "test_spec"
    assert bw_ref.id in ref.parent_ids
