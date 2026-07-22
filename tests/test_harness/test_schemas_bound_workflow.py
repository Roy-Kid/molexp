"""Tests for ``BoundWorkflow`` (``molexp.harness.schemas.bound_workflow``).

Locks the single-definition contract: ``DependencyEdge`` is re-used from
``workflow_ir``, never redefined in the ``bound_workflow`` module.
"""

from __future__ import annotations

from typing import get_args


class TestBoundWorkflow:
    def test_edges_reuse_dependency_edge_from_workflow_ir(self) -> None:
        """``edges`` is ``list[DependencyEdge]`` bound to the IR's type, not a copy."""
        from molexp.harness.schemas.bound_workflow import BoundWorkflow
        from molexp.harness.schemas.workflow_ir import DependencyEdge as IREdge

        field = BoundWorkflow.model_fields["edges"]
        edge_type = get_args(field.annotation)[0]
        assert edge_type is IREdge
