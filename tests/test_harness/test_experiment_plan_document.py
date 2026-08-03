"""Unit tests for the canonical experiment plan document projection."""

from __future__ import annotations

from molexp.harness.plan import (
    BoardTask,
    ExperimentPlan,
    TaskBoard,
    render_experiment_plan_document,
)
from molexp.harness.plan.document import EXPERIMENT_PLAN_OUTLINE, experiment_report_to_document
from molexp.harness.schemas.experiment_report import ExperimentReport


def test_outline_has_twelve_sections() -> None:
    for n, heading in (
        (1, "## 1. Goal"),
        (2, "## 2. Scientific Questions"),
        (7, "## 7. Tasks"),
        (12, "## 12. Deliverables"),
    ):
        assert heading in EXPERIMENT_PLAN_OUTLINE, f"missing section {n}"
    assert "# Experiment Plan" in EXPERIMENT_PLAN_OUTLINE


def test_render_fills_goal_and_tasks() -> None:
    plan = ExperimentPlan(
        spec={"title": "PE Rg", "objective": "Measure Rg vs N"},
        board=TaskBoard(
            tasks=(
                BoardTask(
                    id="t1",
                    name="Build chain",
                    acceptance=("topology written",),
                ),
                BoardTask(
                    id="t2",
                    name="Compute Rg",
                    acceptance=("Rg table",),
                ),
            )
        ),
    )
    md = render_experiment_plan_document(plan)
    assert md.startswith("# Experiment Plan: PE Rg")
    assert "## 1. Goal" in md
    assert "Measure Rg vs N" in md
    assert "### Task 1" in md
    assert "Build chain" in md
    assert "topology written" in md
    assert "### Task 2" in md
    assert "```text" in md
    assert "Build chain" in md and "Compute Rg" in md


def test_experiment_report_body_md_preferred() -> None:
    report = ExperimentReport(
        title="t",
        objective="o",
        system_description="s",
        experimental_design="e",
        body_md="# Experiment Plan: custom\n\n## 1. Goal\n\n**Objective**\n\n> custom body\n",
    )
    assert "custom body" in report.to_document_md()
    assert experiment_report_to_document(report).startswith("# Experiment Plan: custom")


def test_experiment_report_reconstructs_without_body_md() -> None:
    report = ExperimentReport(
        title="Legacy",
        objective="Do science",
        system_description="water box",
        experimental_design="MD then analyze",
    )
    md = report.to_document_md()
    assert "# Experiment Plan: Legacy" in md
    assert "Do science" in md
    assert "## 5. Experimental Design" in md
