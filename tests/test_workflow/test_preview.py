"""Tests for WorkflowPreviewView and render_preview.

WorkflowPreviewView lives in molexp.workflow.preview and is the
agent-facing derived view of a PlanProposal. Crucially it does NOT
hold workflow_ir — the IR belongs to PlanProposal; preview is just
python_script + mermaid + intervention_points.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.workflow import (
    InterventionPoint,
    PlanProposal,
    TaskProposal,
    WorkflowPreviewView,
)
from molexp.workflow.preview import render_preview


def _registered(task_id: str = "t1") -> TaskProposal:
    return TaskProposal(
        task_id=task_id,
        kind="registered",
        task_type="core.add",
    )


class TestWorkflowPreviewViewShape:
    def test_construct_default(self):
        v = WorkflowPreviewView()
        assert v.python_script == ""
        assert v.mermaid == ""
        assert v.intervention_points == ()

    def test_construct_with_fields(self):
        v = WorkflowPreviewView(
            python_script="print(1)",
            mermaid="flowchart TD",
            intervention_points=("ip1", "ip2"),
        )
        assert v.python_script == "print(1)"
        assert v.mermaid == "flowchart TD"
        assert v.intervention_points == ("ip1", "ip2")

    def test_frozen_blocks_mutation(self):
        v = WorkflowPreviewView()
        with pytest.raises(ValidationError):
            v.python_script = "tampered"  # type: ignore[misc]

    def test_no_workflow_ir_field(self):
        """workflow_ir belongs to PlanProposal, not preview."""
        with pytest.raises(ValidationError):
            WorkflowPreviewView(workflow_ir={})  # type: ignore[call-arg]


class TestRenderPreview:
    def test_empty_proposal_yields_empty_view(self):
        plan = PlanProposal(name="p", task_proposals=(_registered(),))
        view = render_preview(plan)
        assert isinstance(view, WorkflowPreviewView)
        assert view.python_script == ""
        assert view.mermaid == ""
        assert view.intervention_points == ()

    def test_intervention_points_are_passed_through(self):
        plan = PlanProposal(
            name="p",
            task_proposals=(_registered(),),
            intervention_points=(
                InterventionPoint(name="user_approval", description="approve plan"),
                InterventionPoint(name="param_check", description="verify"),
            ),
        )
        view = render_preview(plan)
        assert "user_approval" in view.intervention_points
        assert "param_check" in view.intervention_points

    def test_render_is_pure(self):
        """Same proposal → same view, no side effects."""
        plan = PlanProposal(name="p", task_proposals=(_registered(),))
        a = render_preview(plan)
        b = render_preview(plan)
        assert a == b
