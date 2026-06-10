"""End-to-end Phase-5 integration test (ac-009).

Ties the Phase-2 pipeline output to both Phase-5 validators:
- validate_provenance on the experiment_report artifact → passed=True
- validate_test_spec against a hand-built WorkflowIR for the same pipeline → passed=True

Also re-confirms the Phase-5 public surface (ac-003).
"""

from __future__ import annotations

import asyncio
from pathlib import Path


def _run_pipeline_and_return_refs(tmp_path: Path):
    """Re-runs SaveUserPlan + GenerateExperimentReport via StubAgentGateway.
    Returns (ctx, user_plan_ref, report_ref).
    """
    from molexp.harness import (
        FileArtifactStore,
        GenerateExperimentReport,
        HarnessRunContext,
        SaveUserPlan,
        SQLiteArtifactLineageStore,
        SQLiteEventLog,
        StageRunner,
    )
    from molexp.harness.gateways.stub import StubAgentGateway

    db_path = tmp_path / "events.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db_path)
    provenance = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    stub = StubAgentGateway(artifact_store=artifacts)
    stub.register(
        agent_name="experiment_report_writer",
        output={
            "title": "Water NEMD",
            "objective": "Measure ionic mobility",
            "system_description": "SPC/E water in a 3nm cubic box",
            "experimental_design": "3 replicas at 300K with external E field",
        },
        raw_text="<verbatim LLM>",
    )
    ctx = HarnessRunContext(
        run_id="run-p5-e2e",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=provenance,
        agent_gateway=stub,
    )

    runner = StageRunner(ctx)
    user_plan_ref = asyncio.run(runner.run_stage(SaveUserPlan(user_text="Simulate water at 300K")))
    report_ref = asyncio.run(runner.run_stage(GenerateExperimentReport()))
    return ctx, user_plan_ref, report_ref


# ----------------------------------------------------- validate_provenance


def test_phase05_validate_provenance_on_pipeline_output(tmp_path: Path) -> None:
    """The Phase-2 e2e experiment_report must trace back to user_plan."""
    from molexp.harness import validate_provenance

    ctx, _user_plan_ref, report_ref = _run_pipeline_and_return_refs(tmp_path)
    report = validate_provenance(
        report_ref.id,
        artifact_store=ctx.artifact_store,
        lineage_store=ctx.lineage_store,
        root_kind="user_plan",
    )
    assert report.passed is True, f"unexpected violations: {report.violations}"
    assert report.violations == []


def test_phase05_validate_provenance_with_custom_root_kind(tmp_path: Path) -> None:
    """root_kind override: report should also trace to itself (the experiment_report)."""
    from molexp.harness import validate_provenance

    ctx, _user_plan_ref, report_ref = _run_pipeline_and_return_refs(tmp_path)
    # The artifact IS an experiment_report → it counts as root_kind.
    result = validate_provenance(
        report_ref.id,
        artifact_store=ctx.artifact_store,
        lineage_store=ctx.lineage_store,
        root_kind="experiment_report",
    )
    assert result.passed is True


# ------------------------------------------------------ validate_test_spec


def test_phase05_validate_test_spec_against_hand_built_ir(tmp_path: Path) -> None:
    """Construct a TestSpec targeting a hand-built WorkflowIR derived from
    the pipeline's experiment_report; assert validate_test_spec passes."""
    from molexp.harness import (
        DependencyEdge,
        ExpectedOutput,
        ParameterValue,
        TaskIR,
        TestSpec,
        WorkflowIR,
        validate_test_spec,
    )

    _ctx, _user_plan_ref, _report_ref = _run_pipeline_and_return_refs(tmp_path)

    # Hand-built WorkflowIR derived from the pipeline's report — Phase-5
    # doesn't ship the agent stage that synthesizes this automatically.
    build = TaskIR(
        id="build_system",
        name="Pack water",
        purpose="Generate SPC/E configuration",
        task_type="molecule_builder",
        inputs={"n_water": ParameterValue(value=512, source="user_provided")},
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
    ir = WorkflowIR(
        id="wf-water-nemd-p5",
        name="water_nemd_p5",
        objective="Compute ionic mobility",
        inputs={"n_water": ParameterValue(value=512, source="user_provided")},
        tasks=[build, run_md],
        edges=[DependencyEdge(source_task_id="build_system", target_task_id="run_md")],
        expected_outputs=[
            ExpectedOutput(name="trajectory", kind="dataset", description="MD trajectory"),
        ],
    )

    spec = TestSpec(
        id="ts-md-existence",
        name="MD output exists",
        kind="artifact_existence_test",
        target_task_id="run_md",
        description="run_md must produce traj.dcd",
        expected_artifacts=["traj.dcd"],
    )
    report = validate_test_spec(spec, ir=ir)
    assert report.passed is True, f"unexpected violations: {report.violations}"
    assert report.violations == []


# ------------------------------------------------- public-surface invariants


def test_phase05_public_symbols_importable_from_top_level() -> None:
    from molexp.harness import (  # noqa: F401
        TestKind,
        TestResult,
        TestSpec,
        TestStatus,
        validate_provenance,
        validate_test_spec,
    )


def test_phase01_to_phase04_surface_still_intact() -> None:
    """Regression: every Phase-1..4 export still importable."""
    from molexp.harness import (  # noqa: F401
        AgentCallResult,
        AgentCallSpec,
        AgentGateway,
        ArtifactKind,
        ArtifactRef,
        BoundTask,
        BoundWorkflow,
        CapabilityRegistry,
        ExperimentReport,
        InMemoryCapabilityRegistry,
        ToolCapability,
        UserPlan,
        validate_bound_workflow,
        validate_workflow_ir,
    )
