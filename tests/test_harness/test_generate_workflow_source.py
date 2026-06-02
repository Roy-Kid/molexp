"""Tests for the ``GenerateWorkflowSource`` stage (ac-006).

Drives the stage with a ``StubAgentGateway`` registered to return a
known-good ``WorkflowSource`` and asserts:
- fail-fast when ``ctx.agent_gateway`` is None;
- the ``AgentCallSpec`` mirrors ``GenerateTestSpec`` exactly
  (agent_name="workflow_source_writer", input ids, output_schema);
- it persists a ``"workflow_source"`` artifact whose ``parent_ids``
  include the bound-workflow artifact id (lineage).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

# Mirror the schema fixture in test_validate_workflow_source so the
# generated source the stub returns is itself a valid molexp.workflow program.
_VALID_SOURCE = """\
from molexp.workflow import Task, TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.task(depends_on=["load"])
    async def square(ctx: TaskContext) -> list[int]:
        return [x * x for x in ctx.inputs]

    return wf
"""


def _workflow_source_canned() -> dict:
    return {
        "source": _VALID_SOURCE,
        "module_name": "generated_workflow",
        "bound_workflow_id": "bw-x",
        "symbols": ["WorkflowCompiler", "Task", "TaskContext"],
    }


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
        run_id="run-gws",
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
        run_id="run-gws",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
        agent_gateway=stub,
    )


def test_name_and_subclass() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

    assert GenerateWorkflowSource.name == "generate_workflow_source"
    assert issubclass(GenerateWorkflowSource, Stage)


def test_fail_fast_no_gateway(ctx_no_gw) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

    _seed_bw_ref(ctx_no_gw.artifact_store)
    stage = GenerateWorkflowSource()
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx_no_gw))
    assert "agent_gateway" in str(exc.value)


def test_builds_correct_spec(ctx_with_gw) -> None:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec, WorkflowSource
    from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

    bw_ref = _seed_bw_ref(ctx_with_gw.artifact_store)
    real_gw = ctx_with_gw.agent_gateway
    real_gw.register(
        agent_name="workflow_source_writer",
        output=_workflow_source_canned(),
        output_kind="workflow_source",
    )
    captured: list[AgentCallSpec] = []

    class Cap:
        async def call(self, spec: AgentCallSpec) -> AgentCallResult:
            captured.append(spec)
            return await real_gw.call(spec)

    object.__setattr__(ctx_with_gw, "_frozen", False)
    ctx_with_gw.agent_gateway = cast(AgentGateway, Cap())
    object.__setattr__(ctx_with_gw, "_frozen", True)

    asyncio.run(GenerateWorkflowSource().run(ctx_with_gw))
    assert len(captured) == 1
    spec = captured[0]
    assert spec.agent_name == "workflow_source_writer"
    assert spec.input_artifact_ids == [bw_ref.id]
    assert spec.output_schema == WorkflowSource.model_json_schema()


def test_persists_workflow_source_artifact_with_lineage(ctx_with_gw) -> None:
    from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

    bw_ref = _seed_bw_ref(ctx_with_gw.artifact_store)
    ctx_with_gw.agent_gateway.register(
        agent_name="workflow_source_writer",
        output=_workflow_source_canned(),
        output_kind="workflow_source",
    )
    ref = asyncio.run(GenerateWorkflowSource().run(ctx_with_gw))
    assert ref.kind == "workflow_source"
    assert bw_ref.id in ref.parent_ids
