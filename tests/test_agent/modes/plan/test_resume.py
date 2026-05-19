"""Integration and unit tests for PlanMode.resume().

PlanMode.resume() rehydrates a half-finished plan workflow by reading
the latest ``workflow.json`` snapshot and seeding the next
``Workflow.execute(seed_outputs=...)`` call with the already-completed
node outputs. The legacy ``drive_with_repair`` driver is gone — the
workflow's own ``wf.loop`` primitive owns iteration control — so the
test surface is correspondingly smaller: we check that resume reads
the manifest, computes the resume frontier, and passes seed outputs
into the workflow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from molexp.agent.modes.plan._mode import PlanMode, _compute_resume_frontier
from molexp.agent.modes.plan.context import PLAN_PIPELINE_ORDER
from molexp.agent.modes.plan.plan_folder import PlanFolder, PlanManifest
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    IngestReportResult,
    PlanBrief,
    PlanBriefResult,
    ReportDigest,
)
from molexp.workspace import Workspace


# ── resume() raises on empty workspace ──────────────────────────────────────


def test_plan_mode_resume_raises_on_empty_completed_nodes(
    tmp_path: Path,
) -> None:
    """resume() raises ``ValueError`` when the folder has no completed nodes."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="test-plan")))
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
    """resume() returns a configured :class:`PlanMode` when work exists."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="test-plan")))
    result = IngestReportResult(report_path=Path("/tmp/report.md"), report_hash="abc123")
    plan_folder.write_node_result("IngestReport", result)
    plan_folder.checkpoint("IngestReport")

    mode = PlanMode.resume(plan_folder=plan_folder)
    assert isinstance(mode, PlanMode)
    # The resumed mode remembers which nodes are already done.
    assert mode._resume_completed == ("IngestReport",)


# ── PlanTask persists results so resume can find them ──────────────────────


@pytest.mark.asyncio
async def test_plan_task_persists_result_and_checkpoint(
    tmp_path: Path, fake_router: object
) -> None:
    """``PlanTask.execute`` writes the result and checkpoints the manifest."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="test-plan")))
    router = fake_router

    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY
    from molexp.agent.modes.plan.protocols import PlanDeps
    from molexp.agent.modes.plan.tasks import IngestReport
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


# ── Resume only executes non-completed nodes ───────────────────────────────


def test_resume_completed_tracks_finished_nodes(tmp_path: Path) -> None:
    """``_resume_completed`` lists nodes that finished, in pipeline order."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="test-plan")))

    r1 = IngestReportResult(report_path=Path("/tmp/r.md"), report_hash="h1")
    plan_folder.write_node_result("IngestReport", r1)
    plan_folder.checkpoint("IngestReport")

    mode = PlanMode.resume(plan_folder=plan_folder)
    assert mode._resume_completed == ("IngestReport",)
    # The resume frontier (used historically) is the next un-completed node.
    next_node = _compute_resume_frontier(mode._resume_completed, PLAN_PIPELINE_ORDER)
    ingest_idx = PLAN_PIPELINE_ORDER.index("IngestReport")
    assert PLAN_PIPELINE_ORDER.index(next_node) > ingest_idx


# ── Mid-pipeline resume threads seed_outputs into execute() ────────────────


def test_resume_from_mid_pipeline_loads_seed_outputs(tmp_path: Path) -> None:
    """After three completed nodes, resume rehydrates their typed outputs."""
    workspace = Workspace(tmp_path / "ws")
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="test-plan")))

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
    completed_set = set(mode._resume_completed)
    assert completed_set >= {"IngestReport", "DraftReportDigest", "DraftImplementationPlan"}

    seed = plan_folder.load_seed_outputs()
    assert len(seed) == 3
    assert isinstance(seed["IngestReport"], IngestReportResult)
    assert isinstance(seed["DraftReportDigest"], DigestResult)
    assert isinstance(seed["DraftImplementationPlan"], PlanBriefResult)


# ── _compute_resume_frontier unit tests ────────────────────────────────────


def test_compute_resume_frontier_returns_first_uncompleted() -> None:
    completed = ("IngestReport", "DraftReportDigest")
    frontier = _compute_resume_frontier(completed, PLAN_PIPELINE_ORDER)
    assert frontier == "ClarifyMissingInformation"


def test_compute_resume_frontier_empty_when_all_completed() -> None:
    frontier = _compute_resume_frontier(PLAN_PIPELINE_ORDER, PLAN_PIPELINE_ORDER)
    assert frontier == ""


def test_compute_resume_frontier_with_empty_completed() -> None:
    frontier = _compute_resume_frontier((), PLAN_PIPELINE_ORDER)
    assert frontier == PLAN_PIPELINE_ORDER[0]
