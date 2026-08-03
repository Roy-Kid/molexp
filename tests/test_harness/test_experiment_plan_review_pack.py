"""RED unit tests for :func:`build_experiment_plan_review_pack` (spec plan-emergent-05b-guard).

The builder is a deterministic (no-LLM) :class:`~molexp.harness.schemas.ReviewPack`
source for the plan-review gate. Per the spec reconciliation it anchors on the
**pre-approval** ``experiment_plan`` artifact (NOT ``frozen_experiment_plan`` —
freeze is a post-approval consequence, so the frozen artifact does not exist yet
at gate-render), read via ``_require_kind`` exactly like
:func:`build_experiment_spec_review_pack`.

Written before the production symbol exists — importing
``build_experiment_plan_review_pack`` fails RED (ImportError) until it is authored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import StageExecutionError
from molexp.harness.plan import (
    BoardTask,
    Difficulty,
    ExperimentPlan,
    FeasibilityAnnotation,
    TaskBoard,
)
from molexp.harness.stages.review_pack_builders import build_experiment_plan_review_pack
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

_TITLE = "Zwitterion CG"
_OBJECTIVE = "Coarse-grain a zwitterion in explicit water"


def _plan() -> ExperimentPlan:
    task = BoardTask(
        id="t-build",
        name="Build system",
        acceptance=("beads placed", "box sized"),
        feasibility=FeasibilityAnnotation(
            reachable=True, difficulty=Difficulty.TRIVIAL, rationale="grounded"
        ),
    )
    return ExperimentPlan(
        spec={"id": "exp-1", "title": _TITLE, "objective": _OBJECTIVE},
        board=TaskBoard(tasks=(task,)),
    )


def _ctx(tmp_path: Path) -> HarnessRunContext:
    db = tmp_path / "harness.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    return HarnessRunContext(
        run_id="run-plan-review",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=SQLiteEventLog(path=db),
        lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=artifacts),
    )


class TestBuildExperimentPlanReviewPack:
    def test_hard_review_over_the_experiment_plan(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        ref = ctx.artifact_store.put_json(
            kind="experiment_plan",
            obj=_plan().model_dump(mode="json"),
            created_by="test",
            parent_ids=[],
        )

        pack = build_experiment_plan_review_pack(ctx)

        assert pack.policy == "hard"
        assert pack.step_id == "review_plan"
        assert pack.step_title == "Review experiment plan"
        assert pack.decision_options == ["approve", "reject", "revise"]
        assert pack.pack_id == f"pack-experiment-plan-{ref.id}"
        assert any(r.id == ref.id for r in pack.subject_refs)

    def test_form_renders_spec_and_board_summary(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        ctx.artifact_store.put_json(
            kind="experiment_plan",
            obj=_plan().model_dump(mode="json"),
            created_by="test",
            parent_ids=[],
        )

        pack = build_experiment_plan_review_pack(ctx)

        # Title/objective live on the form shell; full plan book on summary_md.
        assert pack.form.title == _TITLE or _TITLE in (pack.form.title or "")
        assert pack.form.description_md is None or _OBJECTIVE in (pack.form.description_md or "")
        assert "# Experiment Plan" in pack.summary_md
        assert "## 1. Goal" in pack.summary_md
        assert "## 7. Tasks" in pack.summary_md
        assert "t-build" in pack.summary_md or "Build system" in pack.summary_md
        assert "beads placed" in pack.summary_md
        assert "feasibility" not in pack.summary_md.lower()

    def test_form_is_comment_only(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        ctx.artifact_store.put_json(
            kind="experiment_plan",
            obj=_plan().model_dump(mode="json"),
            created_by="test",
            parent_ids=[],
        )

        pack = build_experiment_plan_review_pack(ctx)

        assert len(pack.form.fields) == 1
        field = pack.form.fields[0]
        assert field.id == "operator_notes"
        assert field.label == "Comment"
        assert field.required is False
        assert {f.id for f in pack.form.fields}.isdisjoint(
            {"priority", "confirm_plan", "keep_tasks", "plan_summary"}
        )


class TestMissingArtifact:
    def test_missing_experiment_plan_raises_stage_execution_error(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        with pytest.raises(StageExecutionError):
            build_experiment_plan_review_pack(ctx)
