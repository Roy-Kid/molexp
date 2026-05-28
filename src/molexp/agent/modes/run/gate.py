"""The ``approve_execution`` gate — RunMode's binding safety property.

``approve_execution`` is the third and most consequential of the three
:class:`~molexp.agent.modes._planning.ApprovalGate`\\ s. PlanMode owns
``approve_direction``, AuthorMode owns ``approve_materialization``, and
RunMode owns ``approve_execution``: the gate a human must clear before
*any* LLM-authored experiment code is imported or executed.

The gate runs through the harness's unified approval path —
:meth:`~molexp.agent.runtime.AgentHarness.approve` — which fires
the ``before_approval`` hook where a :class:`~molexp.agent.review.ReviewPolicy`
-shaped handler returns the :class:`~molexp.agent.review.ReviewDecision`.
:func:`approve_execution_gate` is the thin wrapper RunMode calls; it
builds the approval *view* and returns the harness's decision.

No ``pydantic_ai`` / ``pydantic_graph`` imports — pure orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.agent.modes._planning import ApprovalGate

if TYPE_CHECKING:
    from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff
    from molexp.agent.review import ReviewDecision
    from molexp.agent.runtime import AgentHarness

__all__ = ["ExecutionApprovalView", "approve_execution_gate"]


class ExecutionApprovalView:
    """The minimal approval view for the ``approve_execution`` gate.

    :meth:`AgentHarness.approve` reads only ``.summary``; this plain
    runtime object satisfies that contract without a pydantic model.

    Attributes:
        summary: One-line description of what is about to be executed.
        plan_id: The plan being executed (for richer reviewers).
        entrypoint: The ``module:symbol`` entrypoint about to be imported.
    """

    def __init__(self, *, summary: str, plan_id: str, entrypoint: str) -> None:
        self.summary = summary
        self.plan_id = plan_id
        self.entrypoint = entrypoint


def build_execution_view(handoff: MaterializedWorkspaceHandoff) -> ExecutionApprovalView:
    """Build the :class:`ExecutionApprovalView` for a materialized handoff."""
    entrypoint = f"{handoff.entrypoint_module}:{handoff.entrypoint_symbol}"
    step_count = len(handoff.plan_graph.steps)
    summary = f"Execute materialized plan {handoff.plan_id}: {step_count} step(s) via {entrypoint}"
    return ExecutionApprovalView(summary=summary, plan_id=handoff.plan_id, entrypoint=entrypoint)


async def approve_execution_gate(
    handoff: MaterializedWorkspaceHandoff,
    *,
    harness: AgentHarness,
) -> ReviewDecision:
    """Consult the ``approve_execution`` gate for a materialized handoff.

    Routes through :meth:`AgentHarness.approve`, which emits
    ``approval_requested`` / ``approval_decided`` and evaluates the
    ``before_approval`` hook. Until the returned
    :class:`~molexp.agent.review.ReviewDecision` is ``approved``, RunMode
    imports and executes nothing.

    Args:
        handoff: The :class:`MaterializedWorkspaceHandoff` to be executed.
        harness: The driving :class:`AgentHarness`.

    Returns:
        The reviewer's :class:`~molexp.agent.review.ReviewDecision`.
    """
    view = build_execution_view(handoff)
    return await harness.approve(ApprovalGate.approve_execution, view)
