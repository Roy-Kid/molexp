"""Tests for the ``GenerateTestCode`` stage (spec ``harness-run-mode-01-substrate``, T03).

Mirrors ``test_generate_workflow_source.py`` exactly:
- fail-fast ``StageExecutionError`` when ``ctx.agent_gateway`` is None;
- the ``AgentCallSpec`` carries ``agent_name="test_code_writer"``, the
  ordered input ids ``[test_spec_id, workflow_source_id]``, and
  ``output_schema == TestSource.model_json_schema()``;
- the returned ref has kind ``"test_source"`` with both input ids in its
  ``parent_ids`` (lineage).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from molexp.harness import ArtifactRef
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.artifact_store import ArtifactStore

_TEST_SOURCE = (
    "from generated_workflow import build_workflow\n"
    "\n"
    "\n"
    "def test_build_workflow_is_callable():\n"
    "    assert callable(build_workflow)\n"
)

_WORKFLOW_SOURCE = """\
from molexp.workflow import Task, TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    return wf
"""


def _test_source_canned() -> dict:
    return {
        "source": _TEST_SOURCE,
        "module_name": "test_generated_workflow",
        "test_spec_id": "ts-001",
        "bound_workflow_id": "bw-x",
        "symbols": [],
    }


def _seed_test_spec(artifact_store: ArtifactStore) -> ArtifactRef:
    from molexp.harness import TestSpec

    spec = TestSpec(
        id="ts-001",
        name="unit: load",
        kind="unit_test",
        target_task_id="load",
        description="the load task emits three integers",
    )
    return artifact_store.put_json(
        kind="test_spec",
        obj=json.loads(spec.model_dump_json()),
        created_by="seed",
        parent_ids=[],
    )


def _seed_workflow_source(artifact_store: ArtifactStore) -> ArtifactRef:
    from molexp.harness import WorkflowSource

    ws = WorkflowSource(
        source=_WORKFLOW_SOURCE,
        module_name="generated_workflow",
        bound_workflow_id="bw-x",
        symbols=("WorkflowCompiler", "Task", "TaskContext"),
    )
    return artifact_store.put_json(
        kind="workflow_source",
        obj=json.loads(ws.model_dump_json()),
        created_by="seed",
        parent_ids=[],
    )


@pytest.fixture()
def ctx_no_gw(tmp_path: Path) -> HarnessRunContext:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-gtc",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


@pytest.fixture()
def ctx_with_gw(tmp_path: Path) -> HarnessRunContext:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    stub = StubAgentGateway(artifact_store=a)
    return HarnessRunContext(
        run_id="run-gtc",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
        agent_gateway=stub,
    )


def test_name_and_subclass() -> None:
    from molexp.harness import GenerateTestCode
    from molexp.harness.core.stage import Stage

    assert GenerateTestCode.name == "generate_test_code"
    assert issubclass(GenerateTestCode, Stage)


def test_fail_fast_no_gateway(ctx_no_gw) -> None:
    from molexp.harness import GenerateTestCode, StageExecutionError

    _seed_test_spec(ctx_no_gw.artifact_store)
    _seed_workflow_source(ctx_no_gw.artifact_store)
    stage = GenerateTestCode()
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx_no_gw))
    assert "agent_gateway" in str(exc.value)


def test_builds_correct_spec(ctx_with_gw) -> None:
    from molexp.harness import GenerateTestCode, TestSource
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec

    ts_ref = _seed_test_spec(ctx_with_gw.artifact_store)
    ws_ref = _seed_workflow_source(ctx_with_gw.artifact_store)
    real_gw = ctx_with_gw.agent_gateway
    real_gw.register(
        agent_name="test_code_writer",
        output=_test_source_canned(),
        output_kind="test_source",
    )
    captured: list[AgentCallSpec] = []

    class Cap:
        async def call(self, spec: AgentCallSpec) -> AgentCallResult:
            captured.append(spec)
            return await real_gw.call(spec)

    object.__setattr__(ctx_with_gw, "_frozen", False)
    ctx_with_gw.agent_gateway = cast(AgentGateway, Cap())
    object.__setattr__(ctx_with_gw, "_frozen", True)

    asyncio.run(GenerateTestCode().run(ctx_with_gw))
    assert len(captured) == 1
    spec = captured[0]
    assert spec.agent_name == "test_code_writer"
    assert spec.input_artifact_ids == [ts_ref.id, ws_ref.id]
    assert spec.output_schema == TestSource.model_json_schema()


def test_persists_test_source_artifact_with_lineage(ctx_with_gw) -> None:
    from molexp.harness import GenerateTestCode

    ts_ref = _seed_test_spec(ctx_with_gw.artifact_store)
    ws_ref = _seed_workflow_source(ctx_with_gw.artifact_store)
    ctx_with_gw.agent_gateway.register(
        agent_name="test_code_writer",
        output=_test_source_canned(),
        output_kind="test_source",
    )
    ref = asyncio.run(GenerateTestCode().run(ctx_with_gw))
    assert ref.kind == "test_source"
    assert ts_ref.id in ref.parent_ids
    assert ws_ref.id in ref.parent_ids
