"""Integration and unit tests for PlanMode.resume() (ac-010 through ac-014).

Covers:
- ac-010: ValueError when no completed_nodes
- ac-011: drive_with_repair initial_spec / initial_seed_outputs parameterisation
- ac-012: PlanTask.execute persists result + checkpoints after step approval
- ac-013: Resumed PlanMode.run executes only non-completed nodes
- ac-014: Mid-pipeline resume produces valid workspace
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from molexp.agent.modes.plan._mode import PlanMode
from molexp.agent.modes.plan.plan_folder import PlanFolder, PlanManifest
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    IngestReportResult,
    PlanBrief,
    PlanBriefResult,
    ReportDigest,
)
from molexp.agent.modes.plan.context import PLAN_PIPELINE_ORDER
from molexp.workspace import Workspace


# ── ac-010: PlanMode.resume raises when no completed_nodes ────────────────


def test_plan_mode_resume_raises_on_empty_completed_nodes(
    tmp_path: Path,
) -> None:
    """ac-010: resume() raises ValueError when PlanFolder has no completed_nodes."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(
        PlanFolder, workspace.add_folder(PlanFolder(name="test-plan"))
    )
    manifest = PlanManifest(
        plan_id="test-plan",
        created_at=datetime.now(tz=UTC),
        report_source="test",
        workflow_ir_path=Path("ir/wf.yaml"),
        completed_nodes=(),
    )
    plan_folder.write_manifest(manifest)

    with pytest.raises(ValueError, match="no completed nodes"):
        PlanMode.resume(plan_folder=plan_folder)


def test_plan_mode_resume_returns_plan_mode_instance(
    tmp_path: Path,
) -> None:
    """ac-010: resume() returns a PlanMode instance when there are completed nodes."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(
        PlanFolder, workspace.add_folder(PlanFolder(name="test-plan"))
    )
    result = IngestReportResult(
        report_path=Path("/tmp/report.md"), report_hash="abc123"
    )
    plan_folder.write_node_result("IngestReport", result)
    plan_folder.checkpoint("IngestReport")

    mode = PlanMode.resume(plan_folder=plan_folder)
    assert isinstance(mode, PlanMode)
    assert mode._resume_from is not None
    assert mode._resume_from != ""


# ── ac-011: drive_with_repair initial_spec / initial_seed_outputs ─────────


@pytest.mark.asyncio
async def test_drive_with_repair_accepts_initial_spec_kwargs(
    tmp_path: Path,
) -> None:
    """ac-011: drive_with_repair accepts and passes through initial_spec / initial_seed_outputs."""
    from molexp.agent.modes.plan._pipeline import PLAN_WORKFLOW
    from molexp.agent.modes.plan._repair_loop import drive_with_repair
    from molexp.agent.modes.plan.protocols import PlanDeps
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY
    from molexp.agent.review import BypassPolicy

    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(
        PlanFolder, workspace.add_folder(PlanFolder(name="test-plan"))
    )

    # Use just the first node as a subgraph — it can run with standard deps
    sub = PLAN_WORKFLOW.subgraph(["IngestReport"], include_downstream=True)

    # We need a router for drive_with_repair; use a minimal one
    class MinimalRouter:
        async def complete_text(self, **kw: object) -> object:
            raise NotImplementedError("not needed for IngestReport")

        async def complete_structured(self, **kw: object) -> object:
            raise NotImplementedError("not needed for IngestReport")

        def clear_usage(self) -> None:
            pass

        def snapshot_usage(self) -> object:
            from molexp.agent.types import UsageBreakdown

            return UsageBreakdown()

    router = MinimalRouter()
    deps = PlanDeps(
        router=router,
        policy=STANDARD_PLAN_POLICY,
        plan_folder=plan_folder,
        step_policy_lookup=lambda: BypassPolicy(),
        final_policy_lookup=lambda: BypassPolicy(),
        capability_probe=None,
        capability_discovery=None,
    )

    result = await drive_with_repair(
        deps,
        "test report content",
        max_iterations=1,
        initial_spec=sub,
        initial_seed_outputs=None,
    )
    # Subgraph only contains IngestReport, which should produce a result
    assert result is not None


# ── ac-012: PlanTask.execute persists result + checkpoint ─────────────────


@pytest.mark.asyncio
async def test_plan_task_persists_result_and_checkpoint(
    tmp_path: Path, fake_router: object
) -> None:
    """ac-012: PlanTask.execute writes result to results/ and calls checkpoint."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(
        PlanFolder, workspace.add_folder(PlanFolder(name="test-plan"))
    )
    router = fake_router

    from molexp.agent.modes.plan.protocols import PlanDeps
    from molexp.agent.modes.plan.tasks import IngestReport
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY
    from molexp.agent.review import BypassPolicy
    from molexp.workflow import TaskContext

    deps = PlanDeps(
        router=router,
        policy=STANDARD_PLAN_POLICY,
        plan_folder=plan_folder,
        step_policy_lookup=lambda: BypassPolicy(),
        final_policy_lookup=lambda: BypassPolicy(),
        capability_probe=None,
        capability_discovery=None,
    )
    task = IngestReport()
    ctx = TaskContext(
        state={},
        deps=deps,
        inputs={},
        config={"user_input": "test report content"},
    )
    await task.execute(ctx)

    result_path = plan_folder.results_dir() / "IngestReport.yaml"
    assert result_path.exists()

    manifest = plan_folder.load_manifest()
    assert "IngestReport" in manifest.completed_nodes


# ── ac-013: Resume only executes non-completed nodes ──────────────────────


def test_resume_frontier_skips_completed_nodes(
    tmp_path: Path,
) -> None:
    """ac-013: resumed PlanMode._resume_from is after all completed nodes."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(
        PlanFolder, workspace.add_folder(PlanFolder(name="test-plan"))
    )

    r1 = IngestReportResult(report_path=Path("/tmp/r.md"), report_hash="h1")
    plan_folder.write_node_result("IngestReport", r1)
    plan_folder.checkpoint("IngestReport")

    mode = PlanMode.resume(plan_folder=plan_folder)
    assert mode._resume_from is not None
    ingest_idx = PLAN_PIPELINE_ORDER.index("IngestReport")
    resume_idx = PLAN_PIPELINE_ORDER.index(mode._resume_from)
    assert resume_idx > ingest_idx


# ── ac-014: Mid-pipeline resume produces valid workspace ──────────────────


def test_resume_from_mid_pipeline_loads_seed_outputs(
    tmp_path: Path,
) -> None:
    """ac-014: after simulating mid-pipeline interrupt, resume rehydrates state."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(
        PlanFolder, workspace.add_folder(PlanFolder(name="test-plan"))
    )

    r1 = IngestReportResult(report_path=Path("/tmp/r.md"), report_hash="h1")
    r2 = DigestResult(
        digest_path=Path("/tmp/d.md"),
        digest=ReportDigest(
            summary="Test summary",
            experimental_goal="Test goal",
        ),
    )
    r3 = PlanBriefResult(
        plan_path=Path("/tmp/p.md"),
        plan_brief=PlanBrief(
            overview="Test overview",
            chosen_method="Test method",
        ),
    )

    plan_folder.write_node_result("IngestReport", r1)
    plan_folder.write_node_result("DraftReportDigest", r2)
    plan_folder.write_node_result("DraftImplementationPlan", r3)
    plan_folder.checkpoint("IngestReport")
    plan_folder.checkpoint("DraftReportDigest")
    plan_folder.checkpoint("DraftImplementationPlan")

    mode = PlanMode.resume(plan_folder=plan_folder)
    completed_set = {"IngestReport", "DraftReportDigest", "DraftImplementationPlan"}
    assert mode._resume_from is not None
    assert mode._resume_from not in completed_set

    seed = plan_folder.load_seed_outputs()
    assert len(seed) == 3
    assert isinstance(seed["IngestReport"], IngestReportResult)
    assert isinstance(seed["DraftReportDigest"], DigestResult)
    assert isinstance(seed["DraftImplementationPlan"], PlanBriefResult)


# ── _compute_resume_frontier ──────────────────────────────────────────────


def test_compute_resume_frontier_returns_first_uncompleted() -> None:
    """Unit test for _compute_resume_frontier."""
    from molexp.agent.modes.plan._mode import _compute_resume_frontier

    completed = ("IngestReport", "DraftReportDigest")
    frontier = _compute_resume_frontier(completed, PLAN_PIPELINE_ORDER)
    assert frontier == "ClarifyMissingInformation"


def test_compute_resume_frontier_empty_when_all_completed() -> None:
    """_compute_resume_frontier returns '' when every node is done."""
    from molexp.agent.modes.plan._mode import _compute_resume_frontier

    frontier = _compute_resume_frontier(PLAN_PIPELINE_ORDER, PLAN_PIPELINE_ORDER)
    assert frontier == ""


def test_compute_resume_frontier_with_empty_completed() -> None:
    """_compute_resume_frontier returns first node when nothing completed."""
    from molexp.agent.modes.plan._mode import _compute_resume_frontier

    frontier = _compute_resume_frontier((), PLAN_PIPELINE_ORDER)
    assert frontier == PLAN_PIPELINE_ORDER[0]
