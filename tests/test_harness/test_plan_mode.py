"""``PlanMode`` offline verification (plan-mode-revival-04).

Deterministic, network-free: a ``StubAgentGateway`` returns canned valid
outputs per planning agent, so ``PlanMode`` runs its full 9-stage sequence on
a tmp ``workspace.Run`` and we assert the ``ModeResult`` + provenance. Live
mode is the ``API_KEY`` flip in ``examples/harness/experiment_pipeline.py``
(offline-canned by default, smoke-gated).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from molexp.harness import Mode, PlanMode
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas import ModeResult, WorkflowSource
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.workspace import Workspace

# A WorkflowSource program that compiles to a real Workflow (public API).
_VALID_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def build_system(ctx: TaskContext) -> dict:
        return {"structure": "system.pdb"}

    @wf.task(depends_on=["build_system"])
    async def simulate(ctx: TaskContext) -> dict:
        return {"trajectory": "traj.dcd"}

    return wf
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
    "source": _VALID_SOURCE,
    "module_name": "water_nemd",
    "bound_workflow_id": "bw-water",
    "symbols": ["WorkflowCompiler", "TaskContext"],
}


def _make_run(tmp_path: Path):
    ws = Workspace(tmp_path / "lab", name="plan-lab")
    ws.materialize()
    return ws.add_project("demo").add_experiment("nemd").add_run(params={})


def _fixture_gateway(run) -> StubAgentGateway:
    # Share the run's artifact dir with the Mode-built ctx store.
    store = FileArtifactStore(root=run.run_dir / "artifacts")
    gw = StubAgentGateway(store)
    gw.register("experiment_report_writer", _EXPERIMENT_REPORT, output_kind="experiment_report")
    gw.register("workflow_ir_extractor", _WORKFLOW_IR, output_kind="workflow_ir")
    gw.register("bound_workflow_binder", _BOUND_WORKFLOW, output_kind="bound_workflow")
    gw.register("workflow_source_writer", _WORKFLOW_SOURCE, output_kind="workflow_source")
    return gw


# ───────────────────────────────────────────── ac-001 / ac-002: shape


def test_plan_mode_is_a_mode_subclass_exported() -> None:
    import molexp.harness as harness

    assert issubclass(PlanMode, Mode)
    assert "PlanMode" in harness.__all__


def test_plan_mode_declares_the_ten_stage_sequence() -> None:
    names = [s.name for s in PlanMode().stages("draft")]
    assert names == [
        "save_user_plan",
        "generate_experiment_report",
        "approve_experiment_spec",  # early review gate, before compilation
        "extract_workflow_ir",
        "validate_workflow_ir",
        "bind_molcrafts_tasks",
        "validate_bound_workflow",
        "generate_workflow_source",
        "validate_workflow_source",
        "approval_gate",  # terminal final_report gate
    ]
    # The review gate sits directly between the report and IR extraction.
    assert names.index("approve_experiment_spec") == names.index("generate_experiment_report") + 1
    assert names.index("extract_workflow_ir") == names.index("approve_experiment_spec") + 1


def test_experiment_spec_is_a_valid_approval_intent() -> None:
    """ac-001 — the new intent validates; the prior intents still do."""
    from datetime import UTC, datetime

    from molexp.harness.schemas import ApprovalRequest

    for intent in (
        "experiment_spec",
        "final_report",
        "hpc_submission",
        "agent_inferred_scientific_parameters",
        "full_execution",
        "large_resource_request",
        "overwrite",
    ):
        req = ApprovalRequest(
            id="x",
            intent=intent,
            reason="r",
            triggered_by_policy="t",
            created_at=datetime.now(tz=UTC),
        )
        assert req.intent == intent


def test_rejecting_approver_aborts_before_workflow_ir(tmp_path: Path) -> None:
    """ac-005 — a rejecting review approver raises before ExtractWorkflowIR, so
    no workflow_ir artifact is ever produced (the report does exist)."""
    from datetime import UTC, datetime

    from molexp.harness import StageExecutionError
    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest

    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run)
    store = FileArtifactStore(root=run.run_dir / "artifacts")

    async def reject(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            request_id=request.id,
            granted=False,
            decided_by="test",
            decided_at=datetime.now(tz=UTC),
            reason="nope",
        )

    try:
        asyncio.run(PlanMode(approver=reject).run(run=run, user_input="draft", gateway=gateway))
    except StageExecutionError:
        pass
    else:  # pragma: no cover - the gate must abort
        raise AssertionError("rejecting approver must raise StageExecutionError")

    assert store.latest_by_kind("experiment_report") is not None
    assert store.latest_by_kind("workflow_ir") is None


def test_default_approver_completes_all_ten_stages(tmp_path: Path) -> None:
    """ac-004 — PlanMode() (auto-grant) runs all ten stages offline."""
    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run)

    result = asyncio.run(PlanMode().run(run=run, user_input="draft", gateway=gateway))
    assert len(result.stage_artifacts) == 10
    kinds = [a.kind for a in result.stage_artifacts]
    assert "workflow_source" in kinds
    assert kinds.count("analysis_result") == 2  # both gates ran to completion


# ───────────────────────────────────── ac-003 / ac-004 / ac-006: offline run


def test_plan_mode_runs_all_stages_offline(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run)

    result = asyncio.run(
        PlanMode().run(run=run, user_input="Simulate NEMD ionic mobility", gateway=gateway)
    )

    assert isinstance(result, ModeResult)
    assert result.mode_name == "plan"
    assert result.run_id == run.id
    kinds = [a.kind for a in result.stage_artifacts]
    # The codegen artifact is present, and the gate cleared (auto-grant).
    assert "workflow_source" in kinds
    assert "analysis_result" in kinds  # ApprovalGate output → ran to completion


def test_plan_mode_produces_recoverable_workflow_source(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run)
    store = FileArtifactStore(root=run.run_dir / "artifacts")

    result = asyncio.run(PlanMode().run(run=run, user_input="draft", gateway=gateway))

    src_refs = [a for a in result.stage_artifacts if a.kind == "workflow_source"]
    assert len(src_refs) == 1
    ws_obj = WorkflowSource.model_validate_json(store.get(src_refs[0].id))
    assert "build_workflow" in ws_obj.source


# ───────────────────────────────────────────── ac-005: provenance lineage


def test_workflow_source_lineage_reaches_user_plan(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    gateway = _fixture_gateway(run)
    store = FileArtifactStore(root=run.run_dir / "artifacts")

    result = asyncio.run(PlanMode().run(run=run, user_input="draft", gateway=gateway))
    src_ref = next(a for a in result.stage_artifacts if a.kind == "workflow_source")
    user_plan_ref = next(a for a in result.stage_artifacts if a.kind == "user_plan")

    provenance = SQLiteArtifactLineageStore(
        path=run.run_dir / "harness.sqlite", artifact_store=store
    )
    ancestors = {ref.id for ref in provenance.trace_backward(src_ref.id)}
    assert user_plan_ref.id in ancestors


# ───────────────────────────────────────── ac-007 / ac-008: live-example guards


def test_flagship_example_imports_without_network_or_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    import examples.harness.experiment_pipeline as pipeline

    assert callable(pipeline.main)  # importing it ran nothing (guarded under __main__)
    assert os.environ.get("DEEPSEEK_API_KEY") is None


def test_examples_dir_has_no_pytest_tests() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples"
    assert not list(examples.rglob("test_*.py"))
