"""End-to-end Phase-8 pipeline test: 7 stages, 8-layer provenance chain."""

from __future__ import annotations

import asyncio
from pathlib import Path


def _experiment_report_canned() -> dict:
    return {
        "title": "Water NEMD",
        "objective": "Measure mobility",
        "system_description": "SPC/E water",
        "experimental_design": "Apply field",
    }


def _workflow_ir_canned() -> dict:
    return {
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


def _bound_workflow_canned() -> dict:
    return {
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


def _test_spec_canned() -> dict:
    return {
        "id": "ts-001",
        "name": "Dry-run sanity",
        "kind": "dry_run_test",
        "description": "Verify build task produces structure.pdb",
        "target_task_id": "build",
        "expected_artifacts": ["structure.pdb"],
    }


def test_seven_stage_pipeline_yields_eight_layer_provenance_chain(tmp_path: Path) -> None:
    from molexp.harness import (
        BindMolcraftsTasks,
        ExtractWorkflowIR,
        FileArtifactStore,
        GenerateExperimentReport,
        GenerateTestSpec,
        HarnessRunContext,
        SaveUserPlan,
        SQLiteArtifactLineageStore,
        SQLiteEventLog,
        StageRunner,
        ValidateBoundWorkflow,
        ValidateWorkflowIR,
    )
    from molexp.harness.gateways.stub import StubAgentGateway

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    stub = StubAgentGateway(artifact_store=a)
    stub.register("experiment_report_writer", _experiment_report_canned())
    stub.register("workflow_ir_extractor", _workflow_ir_canned(), output_kind="workflow_ir")
    stub.register("bound_workflow_binder", _bound_workflow_canned(), output_kind="bound_workflow")
    stub.register("test_spec_writer", _test_spec_canned(), output_kind="test_spec")

    ctx = HarnessRunContext(
        run_id="run-p8",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
        agent_gateway=stub,
    )
    runner = StageRunner(ctx)

    user_plan = asyncio.run(runner.run_stage(SaveUserPlan(user_text="Simulate water")))
    report = asyncio.run(runner.run_stage(GenerateExperimentReport()))
    workflow_ir = asyncio.run(runner.run_stage(ExtractWorkflowIR()))
    ir_validation = asyncio.run(runner.run_stage(ValidateWorkflowIR()))
    bound_wf = asyncio.run(runner.run_stage(BindMolcraftsTasks()))
    bw_validation = asyncio.run(runner.run_stage(ValidateBoundWorkflow()))
    test_spec = asyncio.run(runner.run_stage(GenerateTestSpec()))

    # Sanity kinds
    assert user_plan.kind == "user_plan"
    assert report.kind == "experiment_report"
    assert workflow_ir.kind == "workflow_ir"
    assert ir_validation.kind == "validation_report"
    assert bound_wf.kind == "bound_workflow"
    assert bw_validation.kind == "validation_report"
    assert test_spec.kind == "test_spec"

    # test_spec.id → bound_wf → workflow_ir → experiment_report → user_plan → raw_text
    ancestors = ctx.lineage_store.trace_backward(test_spec.id)
    ids = [r.id for r in ancestors]
    assert bound_wf.id in ids
    assert workflow_ir.id in ids
    assert report.id in ids
    assert user_plan.id in ids
    # raw_text artifact (kind=user_plan with empty parent_ids) is the deepest ancestor.
    deepest = ctx.artifact_store.get_ref(ids[-1])
    assert deepest.kind == "user_plan"
    assert deepest.parent_ids == []


def test_phase08_public_symbols_importable() -> None:
    from molexp.harness import (  # noqa: F401
        BindMolcraftsTasks,
        GenerateTestSpec,
        ValidateBoundWorkflow,
    )
