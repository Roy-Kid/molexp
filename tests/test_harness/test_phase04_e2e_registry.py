"""End-to-end Phase-4 capability-registry integration test (ac-010).

Registers three real-looking capabilities, builds a matched IR + Bound
that uses each one, validates with registry → passed=True. Then flips
fields to surface `unknown_capability` and `undeclared_side_effect`.
"""

from __future__ import annotations

from pathlib import Path


def _three_capabilities():
    from molexp.harness import ToolCapability

    builder = ToolCapability(
        id="molpy.builder.water.SPCEBuilder",
        package="molpy",
        name="SPCEBuilder",
        description="Pack SPC/E water in a cubic box",
        input_schema={
            "type": "object",
            "properties": {
                "n_water_molecules": {"type": "integer"},
                "box_size_nm": {"type": "number"},
            },
            "required": ["n_water_molecules", "box_size_nm"],
        },
        output_schema={"type": "object", "properties": {"structure": {"type": "string"}}},
        supported_backends=["local"],
        side_effects=["fs_write"],
        tags=["builder", "water"],
    )
    runner = ToolCapability(
        id="molpy.md.NEMDRunner",
        package="molpy",
        name="NEMDRunner",
        description="Run non-equilibrium MD under an electric field",
        input_schema={
            "type": "object",
            "properties": {"structure": {"type": "string"}},
            "required": ["structure"],
        },
        output_schema={"type": "object", "properties": {"trajectory": {"type": "string"}}},
        supported_backends=["local"],
        side_effects=["fs_write", "gpu"],
        tags=["md", "nemd"],
    )
    analyzer = ToolCapability(
        id="molpy.analysis.MobilityKernel",
        package="molpy",
        name="MobilityKernel",
        description="Compute ionic mobility from a trajectory",
        input_schema={
            "type": "object",
            "properties": {"trajectory": {"type": "string"}},
            "required": ["trajectory"],
        },
        output_schema={"type": "object", "properties": {"mobility": {"type": "string"}}},
        supported_backends=["local"],
        side_effects=["fs_write"],
        tags=["analysis"],
    )
    return builder, runner, analyzer


def _build_matched_pair():
    from molexp.harness import (
        BoundTask,
        BoundWorkflow,
        DependencyEdge,
        ExecutionEnvironment,
        ExpectedOutput,
        InMemoryCapabilityRegistry,
        ParameterValue,
        ResourcePolicy,
        TaskIR,
        WorkflowIR,
    )

    builder, runner, analyzer = _three_capabilities()
    registry = InMemoryCapabilityRegistry(capabilities=[builder, runner, analyzer])

    build = TaskIR(
        id="build",
        name="Pack water",
        purpose="Generate SPC/E configuration",
        task_type="molecule_builder",
        inputs={
            "n_water_molecules": ParameterValue(value=512, source="user_provided"),
            "box_size_nm": ParameterValue(value=3.0, source="user_provided"),
        },
        outputs={"structure": "structure.pdb"},
    )
    run_md = TaskIR(
        id="run_md",
        name="Run NEMD",
        purpose="Apply external field",
        task_type="md_runner",
        inputs={"structure": ParameterValue(value="structure.pdb", source="user_provided")},
        outputs={"trajectory": "traj.dcd"},
    )
    analyze = TaskIR(
        id="analyze",
        name="Compute mobility",
        purpose="Estimate ionic mobility",
        task_type="analysis",
        inputs={"trajectory": ParameterValue(value="traj.dcd", source="user_provided")},
        outputs={"mobility": "mobility.json"},
    )
    ir = WorkflowIR(
        id="wf-water-nemd",
        name="water_nemd_e2e",
        objective="Compute ionic mobility",
        inputs={
            "n_water_molecules": ParameterValue(value=512, source="user_provided"),
            "box_size_nm": ParameterValue(value=3.0, source="user_provided"),
        },
        tasks=[build, run_md, analyze],
        edges=[
            DependencyEdge(source_task_id="build", target_task_id="run_md"),
            DependencyEdge(source_task_id="run_md", target_task_id="analyze"),
        ],
        expected_outputs=[
            ExpectedOutput(name="mobility", kind="analysis_result", description="Mobility"),
        ],
    )

    b_build = BoundTask(
        id="b-build",
        ir_task_id="build",
        capability_id="molpy.builder.water.SPCEBuilder",
        package="molpy",
        callable="molpy.builder.water.SPCEBuilder.run",
        parameters={
            "n_water_molecules": ParameterValue(value=512, source="user_provided"),
            "box_size_nm": ParameterValue(value=3.0, source="user_provided"),
        },
        inputs={"n_water_molecules": "wf:n_water_molecules", "box_size_nm": "wf:box_size_nm"},
        outputs={"structure": "structure.pdb"},
        side_effects=["fs_write"],
    )
    b_run = BoundTask(
        id="b-run",
        ir_task_id="run_md",
        capability_id="molpy.md.NEMDRunner",
        package="molpy",
        callable="molpy.md.NEMDRunner.run",
        parameters={"structure": ParameterValue(value="structure.pdb", source="user_provided")},
        inputs={"structure": "b-build:structure"},
        outputs={"trajectory": "traj.dcd"},
        side_effects=["fs_write", "gpu"],
    )
    b_analyze = BoundTask(
        id="b-analyze",
        ir_task_id="analyze",
        capability_id="molpy.analysis.MobilityKernel",
        package="molpy",
        callable="molpy.analysis.MobilityKernel.run",
        parameters={"trajectory": ParameterValue(value="traj.dcd", source="user_provided")},
        inputs={"trajectory": "b-run:trajectory"},
        outputs={"mobility": "mobility.json"},
        side_effects=["fs_write"],
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
    return ir, bw, registry


def _codes(report) -> list[str]:
    return [v.code for v in report.violations]


def test_three_capability_pipeline_validates_clean(tmp_path: Path) -> None:
    from molexp.harness import BoundWorkflowValidator

    ir, bw, registry = _build_matched_pair()
    report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path, registry=registry)
    assert report.passed is True, f"unexpected violations: {report.violations}"
    assert report.violations == []


def test_flipping_to_ghost_capability_triggers_unknown_capability(tmp_path: Path) -> None:
    from molexp.harness import BoundWorkflowValidator

    ir, bw, registry = _build_matched_pair()
    bad_run = bw.tasks[1].model_copy(update={"capability_id": "ghost.capability"})
    bw_bad = bw.model_copy(update={"tasks": [bw.tasks[0], bad_run, bw.tasks[2]]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    assert "unknown_capability" in _codes(report)


def test_undeclared_side_effect_fires(tmp_path: Path) -> None:
    from molexp.harness import BoundWorkflowValidator

    ir, bw, registry = _build_matched_pair()
    # MobilityKernel only declares ["fs_write"]; add "network" which it didn't.
    bad_analyze = bw.tasks[2].model_copy(update={"side_effects": ["fs_write", "network"]})
    bw_bad = bw.model_copy(update={"tasks": [bw.tasks[0], bw.tasks[1], bad_analyze]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    assert "undeclared_side_effect" in _codes(report)


def test_phase04_public_surface_complete() -> None:
    from molexp.harness import (  # noqa: F401
        CapabilityAlreadyRegisteredError,
        CapabilityCallValidationError,
        CapabilityNotFoundError,
        CapabilityRegistry,
        InMemoryCapabilityRegistry,
        ToolCapability,
    )


def test_phase01_to_phase03_surface_still_intact() -> None:
    """Regression: every Phase-1/2/3 export must still import."""
    from molexp.harness import (  # noqa: F401
        AgentCallResult,
        AgentCallSpec,
        AgentGateway,
        ArtifactKind,
        ArtifactRef,
        BoundTask,
        BoundWorkflow,
        BoundWorkflowValidator,
        DependencyEdge,
        ExperimentReport,
        FileArtifactStore,
        GenerateExperimentReport,
        HarnessRunContext,
        SaveUserPlan,
        Stage,
        StageRunner,
        TaskIR,
        UserPlan,
        ValidationReport,
        ValidationViolation,
        WorkflowIR,
        WorkflowIRValidator,
    )
