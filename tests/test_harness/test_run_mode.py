"""``RunMode`` offline verification (harness-run-mode-02, T01 + T02).

Deterministic, network-free: a ``StubAgentGateway`` returns canned valid
outputs for all 7 pipeline agents (4 plan + 3 run), so ``PlanMode`` then
``RunMode`` run their full stage sequences against the **same** tmp
``workspace.Run``. The canned workflow source and the canned passing test
source are real programs (verified by hand: the workflow completes under
``WorkflowRuntime`` with ``outputs["summarize"]["total"] == 6``; the pytest
module passes in the materialized ``generated/`` layout) — the default
``LocalExecutor`` therefore spawns genuine pytest + driver subprocesses in
the integration tests. No LLM, no network, no wall-clock assertions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest

from molexp.harness import (
    ApprovalGate,
    ArtifactNotFoundError,
    DryRunExecutor,
    ExecuteTests,
    ExecuteWorkflow,
    ExecutionResult,
    GenerateAuditReport,
    GenerateFinalReport,
    GenerateTestCode,
    GenerateTestSpec,
    MaterializeExecution,
    Mode,
    ModeResult,
    PlanMode,
    RunMode,
    StagePersistedFailureError,
    ValidateTestSource,
    ValidateTestSpec,
)
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.workspace import Workspace

_DRAFT = "Simulate NEMD ionic mobility"

# A WorkflowSource program that compiles AND executes (verified by hand under
# ``WorkflowRuntime``): two tasks, values-on-edges — ``summarize`` receives
# its single upstream's output directly as ``ctx.inputs``.
_RUNNABLE_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="canned_run")

    @wf.task
    async def make_data(ctx: TaskContext) -> dict:
        return {"values": [1, 2, 3]}

    @wf.task(depends_on=["make_data"])
    async def summarize(ctx: TaskContext) -> dict:
        return {"total": sum(ctx.inputs["values"])}

    return wf
"""

# A pytest module that passes in the materialized layout (verified by hand:
# ``python -m pytest test_generated_workflow.py -q`` next to the generated
# ``generated_workflow.py`` exits 0).
_PASSING_TEST_SOURCE = """\
from generated_workflow import build_workflow


def test_build_compiles() -> None:
    assert build_workflow().compile() is not None
"""

_EXPERIMENT_REPORT = {
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "system_description": "SPC/E water box under an applied field",
    "experimental_design": "Apply field; record current",
}
_WORKFLOW_IR = {
    "id": "wf-water",
    "name": "water_nemd",
    "objective": "Compute mobility",
    "inputs": {},
    "tasks": [
        {
            "id": "build",
            "name": "Pack water",
            "purpose": "Build SPC/E box",
            "task_type": "molecule_builder",
            "inputs": {},
            "outputs": {"structure": "structure.pdb"},
        }
    ],
    "edges": [],
    "expected_outputs": [],
}
_BOUND_WORKFLOW = {
    "id": "bw-water",
    "workflow_ir_id": "wf-water",
    "tasks": [
        {
            "id": "b-build",
            "ir_task_id": "build",
            "capability_id": "molpy.builder.water.SPCEBuilder",
            "package": "molpy",
            "callable": "molpy.builder.water.SPCEBuilder.run",
            "parameters": {},
            "inputs": {},
            "outputs": {"structure": "structure.pdb"},
        }
    ],
    "edges": [],
    "execution_backend": "local",
    "environment": {},
    "resource_policy": {
        "backend": "local",
        "max_runtime_s": 3600,
        "denied_paths": ["/", "~/.ssh"],
    },
}
_WORKFLOW_SOURCE = {
    "source": _RUNNABLE_SOURCE,
    "module_name": "generated_workflow",
    "bound_workflow_id": "bw-water",
    "symbols": ["WorkflowCompiler", "TaskContext"],
}
# One TestSpec per BoundTask, bundled into the single test_spec artifact. The
# lone BoundTask "build" → the generated source must carry a test covering it.
_TEST_SPEC = {
    "id": "tsb-water",
    "bound_workflow_id": "bw-water",
    "specs": [
        {
            "id": "ts-water",
            "name": "workflow_compiles",
            "kind": "unit_test",
            "target_task_id": "build",
            "description": "The generated workflow module compiles into a workflow.",
        }
    ],
}
_TEST_SOURCE = {
    "source": _PASSING_TEST_SOURCE,
    "module_name": "test_generated_workflow",
    "test_spec_id": "ts-water",
    "bound_workflow_id": "bw-water",
    "symbols": ["build_workflow"],
}
# Structurally valid (covers task "build") but red at pytest runtime, so the
# block happens at ExecuteTests, not ValidateTestSource.
_FAILING_TEST_SOURCE = {
    **_TEST_SOURCE,
    "source": "def test_build_runs() -> None:\n    assert False\n",
}
_FINAL_REPORT = {
    "title": "CannedWaterNemdFinalReport",
    "objective": "Measure ionic mobility from real execution outputs.",
    "methods_summary": "Two-task canned workflow executed by the harness driver.",
    "test_summary": "Generated unit test compiled the workflow and passed.",
    "execution_summary": "Driver subprocess exited 0; outputs.json collected.",
    "results": "summarize reported total == 6 from values [1, 2, 3].",
    "conclusions": "Run-mode wiring carries plan artifacts to real execution.",
    "limitations": ["single canned seed"],
    "next_steps": ["sweep field strengths"],
}


def _make_run(tmp_path: Path):
    ws = Workspace(tmp_path / "lab", name="run-lab")
    ws.materialize()
    return ws.add_project("demo").add_experiment("nemd").add_run(params={})


def _fixture_gateway(run, *, test_source: Mapping[str, object] | None = None) -> StubAgentGateway:
    """StubAgentGateway with canned outputs for all 7 pipeline agents."""
    # Share the run's artifact dir with the Mode-built ctx store.
    store = FileArtifactStore(root=run.run_dir / "artifacts")
    gw = StubAgentGateway(store)
    gw.register("experiment_report_writer", _EXPERIMENT_REPORT, output_kind="experiment_report")
    gw.register("workflow_ir_extractor", _WORKFLOW_IR, output_kind="workflow_ir")
    gw.register("bound_workflow_binder", _BOUND_WORKFLOW, output_kind="bound_workflow")
    gw.register("workflow_source_writer", _WORKFLOW_SOURCE, output_kind="workflow_source")
    gw.register("test_spec_writer", _TEST_SPEC, output_kind="test_spec")
    gw.register("test_code_writer", dict(test_source or _TEST_SOURCE), output_kind="test_source")
    gw.register("final_report_writer", _FINAL_REPORT, output_kind="final_report")
    return gw


# ──────────────────────────────────────────── T01: shape / ctor / sequence


def test_run_mode_is_a_mode_subclass_exported() -> None:
    import molexp.harness as harness

    assert issubclass(RunMode, Mode)
    assert RunMode.name == "run"
    assert "RunMode" in harness.__all__


def test_run_mode_constructs_with_default_and_injected_executor() -> None:
    assert isinstance(RunMode(), RunMode)  # default executor: LocalExecutor()
    assert isinstance(RunMode(executor=DryRunExecutor()), RunMode)


def test_run_mode_declares_the_ten_stage_class_sequence() -> None:
    stage_types = [type(stage) for stage in RunMode().stages("draft")]
    assert stage_types == [
        GenerateTestSpec,
        ValidateTestSpec,
        GenerateTestCode,
        ValidateTestSource,
        MaterializeExecution,
        ExecuteTests,
        ExecuteWorkflow,
        GenerateFinalReport,
        ApprovalGate,
        GenerateAuditReport,
    ]


# ──────────────────────────────────────────── T01: offline e2e happy path


@pytest.mark.integration
def test_run_mode_executes_planned_workflow_end_to_end(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run)
    store = FileArtifactStore(root=run.run_dir / "artifacts")

    plan_result = asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))
    result = asyncio.run(RunMode().run(run=run, user_input=_DRAFT, gateway=gateway))

    assert isinstance(result, ModeResult)
    assert result.mode_name == "run"
    assert result.run_id == run.id
    kinds = {a.kind for a in result.stage_artifacts}
    assert kinds >= {
        "test_spec",
        "validation_report",
        "test_source",
        "test_result",
        "execution_result",
        "final_report",
        "analysis_result",
        "audit_report",
    }

    # The driver really ran the canned workflow: outputs.json round-trips.
    exec_ref = next(a for a in result.stage_artifacts if a.kind == "execution_result")
    execution = ExecutionResult.model_validate_json(store.get(exec_ref.id))
    assert execution.status == "succeeded"
    assert execution.outputs["summarize"]["total"] == 6

    # Cross-mode lineage: the final report's ancestry reaches plan artifacts.
    final_ref = next(a for a in result.stage_artifacts if a.kind == "final_report")
    provenance = SQLiteArtifactLineageStore(
        path=run.run_dir / "harness.sqlite", artifact_store=store
    )
    ancestors = {ref.id for ref in provenance.trace_backward(final_ref.id)}
    plan_ids = {
        a.id
        for a in plan_result.stage_artifacts
        if a.kind in {"experiment_report", "workflow_source"}
    }
    assert ancestors & plan_ids, "final_report ancestry must reach plan-mode artifacts"


# ──────────────────────────────────────── T02: red tests block execution


@pytest.mark.integration
def test_failing_generated_tests_block_workflow_execution(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run, test_source=_FAILING_TEST_SOURCE)
    store = FileArtifactStore(root=run.run_dir / "artifacts")

    asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))
    with pytest.raises(StagePersistedFailureError):
        asyncio.run(RunMode().run(run=run, user_input=_DRAFT, gateway=gateway))

    # In-function import: a module-level `TestResult` would be collected by
    # pytest as a test class (house pattern for Test*-named schemas).
    from molexp.harness import TestResult

    test_ref = store.latest_by_kind("test_result")
    assert test_ref is not None, "TestResult must be persisted before the stage raises"
    test_result = TestResult.model_validate_json(store.get(test_ref.id))
    assert test_result.status == "failed"
    assert store.latest_by_kind("execution_result") is None


# ──────────────────────────────────────────── T02: plan-less Run pre-guard


def test_run_mode_without_plan_artifacts_names_the_remedy(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    gateway = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))

    with pytest.raises(ArtifactNotFoundError, match="molexp plan"):
        asyncio.run(RunMode().run(run=run, user_input=_DRAFT, gateway=gateway))


# ──────────────────────────────────────────────── T02: ledger resume


@pytest.mark.integration
def test_second_run_reuses_ledger_with_unregistered_gateway(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run)

    asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))
    first = asyncio.run(RunMode().run(run=run, user_input=_DRAFT, gateway=gateway))

    # Nothing registered: any re-run of an LLM stage body would raise
    # AgentResponseNotRegisteredError, so completing proves every stage
    # was skipped via the per-run completion ledger.
    empty_gateway = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
    second = asyncio.run(RunMode().run(run=run, user_input=_DRAFT, gateway=empty_gateway))

    assert [a.id for a in second.stage_artifacts] == [a.id for a in first.stage_artifacts]
