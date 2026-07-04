"""Tests for BoundWorkflow (Phase 3 §4.7).

Locks the single-definition contract: DependencyEdge is re-used from
workflow_ir, never redefined in the bound_workflow module.
"""

from __future__ import annotations


def test_bound_workflow_edges_uses_dependency_edge_from_workflow_ir() -> None:
    """Same DependencyEdge type as PlanWorkflowIR — not re-defined."""
    from molexp.harness.schemas.bound_workflow import BoundWorkflow
    from molexp.harness.schemas.workflow_ir import DependencyEdge as IREdge

    field = BoundWorkflow.model_fields["edges"]
    # The annotation is list[DependencyEdge]; type origin = list, args[0] = IREdge.
    from typing import get_args

    edge_type = get_args(field.annotation)[0]
    assert edge_type is IREdge
