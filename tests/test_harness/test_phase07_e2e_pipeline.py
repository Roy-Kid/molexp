"""End-to-end Phase-7 pipeline test (ac-011).

Runs a representative five-stage planning chain:
  SaveUserPlan → GenerateExperimentReport → GenerateExperimentSpec
  → ExtractWorkflowIR → ValidateWorkflowIR
through a single StageRunner with a StubAgentGateway registered for the three
LLM agent names. Verifies:
- Five ArtifactRefs returned, ending in [.., experiment_spec, workflow_ir, validation_report]
- Provenance chain: validation_report ← workflow_ir ← experiment_spec ← experiment_report ← user_plan ← raw_text
- Event log carries one stage-bracket quartet per stage
"""

from __future__ import annotations

import asyncio
from pathlib import Path


def _experiment_spec_canned() -> dict:
    return {
        "id": "spec-water",
        "experiment_report_id": "rep-water",
        "title": "Water NEMD",
        "objective": "Measure ionic mobility",
    }


def _workflow_ir_canned_response() -> dict:
    return {
        "id": "wf-water-nemd",
        "name": "water_nemd",
        "objective": "Compute ionic mobility under field",
        "inputs": {},
        "tasks": [
            {
                "id": "build",
                "name": "Pack water",
                "purpose": "Generate SPC/E configuration",
                "task_type": "molecule_builder",
                "inputs": {},
                "outputs": {"structure": "structure.pdb"},
            }
        ],
        "edges": [],
        "expected_outputs": [],
    }


class TestPhase07E2EPipeline:
    def test_five_stage_pipeline_yields_provenance_chain(self, tmp_path: Path) -> None:
        from molexp.harness import (
            ExtractWorkflowIR,
            FileArtifactStore,
            GenerateExperimentReport,
            GenerateExperimentSpec,
            HarnessRunContext,
            SaveUserPlan,
            SQLiteArtifactLineageStore,
            SQLiteEventLog,
            StageRunner,
            ValidateWorkflowIR,
        )
        from molexp.harness.gateways.stub import StubAgentGateway

        # Wire ctx + stub gateway with three canned responses.
        db = tmp_path / "events.sqlite"
        artifacts = FileArtifactStore(root=tmp_path / "artifacts")
        events = SQLiteEventLog(path=db)
        provenance = SQLiteArtifactLineageStore(path=db, artifact_store=artifacts)
        stub = StubAgentGateway(artifact_store=artifacts)
        stub.register(
            agent_name="experiment_report_writer",
            output={
                "title": "Water NEMD",
                "objective": "Measure ionic mobility",
                "system_description": "SPC/E water box",
                "experimental_design": "Apply external E field",
            },
            raw_text="<verbatim>",
        )
        stub.register(
            agent_name="experiment_spec_generator",
            output=_experiment_spec_canned(),
            output_kind="experiment_spec",
            raw_text="<verbatim>",
        )
        stub.register(
            agent_name="workflow_ir_extractor",
            output=_workflow_ir_canned_response(),
            output_kind="workflow_ir",
            raw_text="<verbatim>",
        )
        ctx = HarnessRunContext(
            run_id="run-p7-e2e",
            workspace_root=tmp_path,
            artifact_store=artifacts,
            event_log=events,
            lineage_store=provenance,
            agent_gateway=stub,
        )
        runner = StageRunner(ctx)

        # Drive the five-stage pipeline.
        user_plan_ref = asyncio.run(
            runner.run_stage(SaveUserPlan(user_text="Simulate water at 300K"))
        )
        report_ref = asyncio.run(runner.run_stage(GenerateExperimentReport()))
        spec_ref = asyncio.run(runner.run_stage(GenerateExperimentSpec()))
        workflow_ir_ref = asyncio.run(runner.run_stage(ExtractWorkflowIR()))
        validation_ref = asyncio.run(runner.run_stage(ValidateWorkflowIR()))

        assert user_plan_ref.kind == "user_plan"
        assert report_ref.kind == "experiment_report"
        assert spec_ref.kind == "experiment_spec"
        assert workflow_ir_ref.kind == "workflow_ir"
        assert validation_ref.kind == "validation_report"

        # Chain: validation_report → workflow_ir → experiment_spec → experiment_report
        #        → user_plan → raw_text
        ancestors = ctx.lineage_store.trace_backward(validation_ref.id)
        assert len(ancestors) == 5
        ids_in_order = [r.id for r in ancestors]
        assert ids_in_order[0] == workflow_ir_ref.id
        assert ids_in_order[1] == spec_ref.id
        assert ids_in_order[2] == report_ref.id
        assert ids_in_order[3] == user_plan_ref.id
        raw_ref = ctx.artifact_store.get_ref(ids_in_order[4])
        assert raw_ref.kind == "user_plan"  # raw text + structured both share the user_plan kind
        assert raw_ref.parent_ids == []

    def test_event_log_contains_one_quartet_per_stage(self, tmp_path: Path) -> None:
        from molexp.harness import (
            ExtractWorkflowIR,
            FileArtifactStore,
            GenerateExperimentReport,
            GenerateExperimentSpec,
            HarnessRunContext,
            SaveUserPlan,
            SQLiteArtifactLineageStore,
            SQLiteEventLog,
            StageRunner,
            ValidateWorkflowIR,
        )
        from molexp.harness.gateways.stub import StubAgentGateway

        db = tmp_path / "events.sqlite"
        artifacts = FileArtifactStore(root=tmp_path / "artifacts")
        events = SQLiteEventLog(path=db)
        provenance = SQLiteArtifactLineageStore(path=db, artifact_store=artifacts)
        stub = StubAgentGateway(artifact_store=artifacts)
        stub.register(
            agent_name="experiment_report_writer",
            output={
                "title": "t",
                "objective": "o",
                "system_description": "s",
                "experimental_design": "e",
            },
        )
        stub.register(
            agent_name="experiment_spec_generator",
            output=_experiment_spec_canned(),
            output_kind="experiment_spec",
        )
        stub.register(
            agent_name="workflow_ir_extractor",
            output=_workflow_ir_canned_response(),
            output_kind="workflow_ir",
        )
        ctx = HarnessRunContext(
            run_id="run-p7-events",
            workspace_root=tmp_path,
            artifact_store=artifacts,
            event_log=events,
            lineage_store=provenance,
            agent_gateway=stub,
        )
        runner = StageRunner(ctx)
        asyncio.run(runner.run_stage(SaveUserPlan(user_text="hi")))
        asyncio.run(runner.run_stage(GenerateExperimentReport()))
        asyncio.run(runner.run_stage(GenerateExperimentSpec()))
        asyncio.run(runner.run_stage(ExtractWorkflowIR()))
        asyncio.run(runner.run_stage(ValidateWorkflowIR()))

        types = [e.type for e in events.list_events("run-p7-events")]
        # One quartet per stage: [stage_started, artifact_created, stage_completed].
        expected_per_stage = ["stage_started", "artifact_created", "stage_completed"]
        assert types == expected_per_stage * 5

    # ---------------------------------- public-surface invariants

    def test_phase07_public_symbols_importable(self) -> None:
        from molexp.harness import ExtractWorkflowIR, ValidateWorkflowIR  # noqa: F401

    def test_phase01_to_phase06_surface_still_intact(self) -> None:
        """Regression: every Phase-1..6 export still resolves."""
        from molexp.harness import (  # noqa: F401
            ApprovalPolicy,
            ApprovalPolicyEvaluator,
            BoundWorkflow,
            BoundWorkflowValidator,
            CapabilityRegistry,
            ExperimentReport,
            GenerateExperimentReport,
            InMemoryCapabilityRegistry,
            ProvenanceValidator,
            SaveUserPlan,
            TestSpecValidator,
            ToolCapability,
            WorkflowIR,
            WorkflowIRValidator,
        )
