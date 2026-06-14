"""Tests for the ``GenerateFinalReport`` stage (spec ``harness-run-mode-01-substrate``, T08).

Mirrors the ``GenerateWorkflowSource`` capture pattern
(``tests/test_harness/test_generate_workflow_source.py``):
- fail-fast ``StageExecutionError`` when ``ctx.agent_gateway`` is None;
- the ``AgentCallSpec`` carries ``agent_name="final_report_writer"``, the
  three upstream artifact ids in order (experiment_report, test_result,
  execution_result — latest of each kind) and
  ``FinalReport.model_json_schema()``;
- returns the gateway's output artifact (kind ``"final_report"``) whose
  ``parent_ids`` include all three inputs.

``ExperimentReport`` / ``TestResult`` exist today; ``ExecutionResult`` and
``FinalReport`` are new in this spec leg — their imports failing IS the
intended RED.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest


def _final_report_canned() -> dict:
    return {
        "title": "Diffusion estimate for an LJ fluid",
        "objective": "Estimate the diffusion coefficient",
        "methods_summary": "Single-task generated workflow",
        "test_summary": "1 generated test, passed",
        "execution_summary": "driver exited 0",
        "results": "estimate_d = 0.25",
        "conclusions": "The estimate is plausible.",
        "limitations": [],
        "next_steps": [],
    }


def _seed_inputs(store):
    from molexp.harness import ExecutionResult, ExperimentReport, TestResult

    report = ExperimentReport(
        title="Diffusion of an LJ fluid",
        objective="Estimate D",
        system_description="LJ particles in a periodic box",
        experimental_design="One MD run, then an MSD fit",
    )
    test_result = TestResult(id="test-result-1", test_spec_id="ts-1", status="passed")
    execution_result = ExecutionResult(
        id="exec-result-1",
        bound_workflow_id="bw-1",
        status="succeeded",
        exit_code=0,
        started_at=datetime(2026, 6, 10, tzinfo=UTC),
        ended_at=datetime(2026, 6, 10, tzinfo=UTC),
        outputs={"estimate_d": 0.25},
        output_artifacts=[],
        stdout=None,
        stderr=None,
        metadata={},
    )
    er_ref = store.put_json(
        kind="experiment_report",
        obj=json.loads(report.model_dump_json()),
        created_by="seed",
        parent_ids=[],
    )
    tr_ref = store.put_json(
        kind="test_result",
        obj=json.loads(test_result.model_dump_json()),
        created_by="seed",
        parent_ids=[],
    )
    xr_ref = store.put_json(
        kind="execution_result",
        obj=json.loads(execution_result.model_dump_json()),
        created_by="seed",
        parent_ids=[],
    )
    return er_ref, tr_ref, xr_ref


@pytest.fixture()
def ctx_no_gw(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-gfr",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


@pytest.fixture()
def ctx_with_gw(tmp_path: Path):
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
        run_id="run-gfr",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
        agent_gateway=stub,
    )


def test_name_and_subclass() -> None:
    from molexp.harness import GenerateFinalReport, Stage

    assert GenerateFinalReport.name == "generate_final_report"
    assert issubclass(GenerateFinalReport, Stage)


def test_fail_fast_no_gateway(ctx_no_gw) -> None:
    from molexp.harness import GenerateFinalReport, StageExecutionError

    _seed_inputs(ctx_no_gw.artifact_store)
    stage = GenerateFinalReport()
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx_no_gw))
    assert "agent_gateway" in str(exc.value)


def test_builds_correct_spec(ctx_with_gw) -> None:
    from molexp.harness import FinalReport, GenerateFinalReport
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec

    er_ref, tr_ref, xr_ref = _seed_inputs(ctx_with_gw.artifact_store)
    real_gw = ctx_with_gw.agent_gateway
    real_gw.register(
        agent_name="final_report_writer",
        output=_final_report_canned(),
        output_kind="final_report",
    )
    captured: list[AgentCallSpec] = []

    class Cap:
        async def call(self, spec: AgentCallSpec) -> AgentCallResult:
            captured.append(spec)
            return await real_gw.call(spec)

    object.__setattr__(ctx_with_gw, "_frozen", False)
    ctx_with_gw.agent_gateway = cast(AgentGateway, Cap())
    object.__setattr__(ctx_with_gw, "_frozen", True)

    asyncio.run(GenerateFinalReport().run(ctx_with_gw))

    assert len(captured) == 1
    spec = captured[0]
    assert spec.agent_name == "final_report_writer"
    assert spec.input_artifact_ids == [er_ref.id, tr_ref.id, xr_ref.id]
    assert spec.output_schema == FinalReport.model_json_schema()


def test_returns_final_report_ref_with_lineage(ctx_with_gw) -> None:
    from molexp.harness import GenerateFinalReport

    er_ref, tr_ref, xr_ref = _seed_inputs(ctx_with_gw.artifact_store)
    ctx_with_gw.agent_gateway.register(
        agent_name="final_report_writer",
        output=_final_report_canned(),
        output_kind="final_report",
    )
    ref = asyncio.run(GenerateFinalReport().run(ctx_with_gw))

    assert ref.kind == "final_report"
    assert er_ref.id in ref.parent_ids
    assert tr_ref.id in ref.parent_ids
    assert xr_ref.id in ref.parent_ids
