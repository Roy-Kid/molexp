"""End-to-end Phase-3 validator test (ac-008).

Hand-constructs a matched WorkflowIR + BoundWorkflow rooted in a
Phase-2-style ExperimentReport sketch, validates both, then mutates each
to trigger one targeted code and asserts the validators catch the defect.

Also re-confirms the Phase-3 public-surface delta (ac-009).
"""

from __future__ import annotations

from pathlib import Path


def _build_matched_pair():
    from molexp.harness import (
        BoundTask,
        BoundWorkflow,
        DependencyEdge,
        ExecutionEnvironment,
        ExpectedOutput,
        ParameterValue,
        ResourcePolicy,
        TaskIR,
        WorkflowIR,
    )

    # Phase-2-style ExperimentReport outline gives us this conceptual chain:
    # build_system → run_md → analyze_trajectory
    build = TaskIR(
        id="build_system",
        name="Pack water box",
        purpose="Generate atomistic SPC/E water configuration",
        task_type="molecule_builder",
        inputs={
            "n_water_molecules": ParameterValue(value=512, source="user_provided"),
            "box_size_nm": ParameterValue(value=3.0, source="user_provided"),
        },
        outputs={"structure": "structure.pdb"},
    )
    run = TaskIR(
        id="run_md",
        name="Run NEMD",
        purpose="Propagate dynamics under external electric field",
        task_type="md_runner",
        inputs={"structure": ParameterValue(value="structure.pdb", source="user_provided")},
        outputs={"trajectory": "traj.dcd"},
    )
    analyze = TaskIR(
        id="analyze_trajectory",
        name="Compute mobility",
        purpose="Estimate ionic mobility from trajectory",
        task_type="analysis",
        inputs={"trajectory": ParameterValue(value="traj.dcd", source="user_provided")},
        outputs={"mobility": "mobility.json"},
    )
    ir = WorkflowIR(
        id="wf-water-nemd",
        name="water_nemd_e2e",
        objective="Compute ionic mobility under field",
        inputs={
            "n_water_molecules": ParameterValue(value=512, source="user_provided"),
            "box_size_nm": ParameterValue(value=3.0, source="user_provided"),
        },
        tasks=[build, run, analyze],
        edges=[
            DependencyEdge(source_task_id="build_system", target_task_id="run_md"),
            DependencyEdge(source_task_id="run_md", target_task_id="analyze_trajectory"),
        ],
        expected_outputs=[
            ExpectedOutput(name="mobility", kind="analysis_result", description="Mobility result"),
        ],
    )

    b_build = BoundTask(
        id="b-build",
        ir_task_id="build_system",
        capability_id="molpy.builder.water.SPCEBuilder",
        package="molpy",
        callable="molpy.builder.water.SPCEBuilder.run",
        parameters={
            "n_water_molecules": ParameterValue(value=512, source="user_provided"),
            "box_size_nm": ParameterValue(value=3.0, source="user_provided"),
        },
        inputs={"n_water_molecules": "wf:n_water_molecules", "box_size_nm": "wf:box_size_nm"},
        outputs={"structure": "structure.pdb"},
    )
    b_run = BoundTask(
        id="b-run",
        ir_task_id="run_md",
        capability_id="molpy.md.NEMDRunner",
        package="molpy",
        callable="molpy.md.NEMDRunner.run",
        parameters={},
        inputs={"structure": "b-build:structure"},
        outputs={"trajectory": "traj.dcd"},
    )
    b_analyze = BoundTask(
        id="b-analyze",
        ir_task_id="analyze_trajectory",
        capability_id="molpy.analysis.MobilityKernel",
        package="molpy",
        callable="molpy.analysis.MobilityKernel.run",
        parameters={},
        inputs={"trajectory": "b-run:trajectory"},
        outputs={"mobility": "mobility.json"},
    )
    bw = BoundWorkflow(
        id="bw-water-nemd",
        workflow_ir_id="wf-water-nemd",
        tasks=[b_build, b_run, b_analyze],
        edges=[
            DependencyEdge(source_task_id="b-build", target_task_id="b-run"),
            DependencyEdge(source_task_id="b-run", target_task_id="b-analyze"),
        ],
        execution_backend="local",
        environment=ExecutionEnvironment(python_version="3.12"),
        resource_policy=ResourcePolicy(
            backend="local",
            max_runtime_s=3600,
            allowed_paths=["./run"],
            denied_paths=["/", "~/.ssh"],
        ),
    )
    return ir, bw


class TestPhase03E2EValidators:
    def test_clean_pair_validates_clean(self, tmp_path: Path) -> None:
        from molexp.harness import BoundWorkflowValidator, WorkflowIRValidator

        ir, bw = _build_matched_pair()
        ir_report = WorkflowIRValidator.validate(ir)
        bw_report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert ir_report.passed, f"IR violations: {ir_report.violations}"
        assert bw_report.passed, f"BW violations: {bw_report.violations}"
        assert ir_report.violations == []
        assert bw_report.violations == []

    def test_drop_edge_target_triggers_unknown_edge_target(self, tmp_path: Path) -> None:
        from molexp.harness import DependencyEdge, WorkflowIRValidator

        ir, _ = _build_matched_pair()
        # Drop run_md's edge target by repointing it at a ghost.
        bad_edges = [
            DependencyEdge(source_task_id="build_system", target_task_id="run_md"),
            DependencyEdge(source_task_id="run_md", target_task_id="ghost_target"),
        ]
        ir_bad = ir.model_copy(update={"edges": bad_edges})
        report = WorkflowIRValidator.validate(ir_bad)
        codes = [v.code for v in report.violations]
        assert "unknown_edge_target" in codes
        assert report.passed is False

    def test_flip_bound_ir_task_id_triggers_unknown_ir_task(self, tmp_path: Path) -> None:
        from molexp.harness import BoundWorkflowValidator

        ir, bw = _build_matched_pair()
        bad_run = bw.tasks[1].model_copy(update={"ir_task_id": "ghost_ir_task"})
        bw_bad = bw.model_copy(update={"tasks": [bw.tasks[0], bad_run, bw.tasks[2]]})
        report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path)
        codes = [v.code for v in report.violations]
        assert "unknown_ir_task" in codes

    # ---------------------------------------------- Public-surface invariants

    def test_phase03_public_surface_complete(self) -> None:
        from molexp.harness import (  # noqa: F401
            BoundTask,
            BoundWorkflow,
            BoundWorkflowValidator,
            DependencyEdge,
            ExecutionEnvironment,
            ExpectedOutput,
            ResourcePolicy,
            TaskIR,
            ValidationReport,
            ValidationViolation,
            WorkflowIR,
            WorkflowIRValidator,
        )

    def test_phase01_and_phase02_surface_still_visible(self) -> None:
        """Regression: existing Phase-1+2 exports must still import cleanly."""
        from molexp.harness import (  # noqa: F401
            AgentCallResult,
            AgentCallSpec,
            AgentGateway,
            ArtifactKind,
            ArtifactRef,
            ArtifactStore,
            EventLog,
            EventType,
            ExperimentReport,
            FileArtifactStore,
            GenerateExperimentReport,
            HarnessError,
            HarnessRunContext,
            ParameterValue,
            SaveUserPlan,
            Stage,
            StageRunner,
            UserPlan,
        )
