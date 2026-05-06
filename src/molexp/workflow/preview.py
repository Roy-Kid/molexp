"""WorkflowPreviewView — derived view of a PlanProposal.

A read-only view of a workflow plan, suitable for surfacing to the
agent / UI / SSE pipeline. **Does not hold the workflow_ir** —
that lives on :class:`PlanProposal` (the source of truth). The
preview is what you'd send into a chat reply or render in a side
panel: a Python script form, a Mermaid diagram, and the names of
intervention points.

The agent layer imports :class:`WorkflowPreviewView` for typing
purposes only — it never constructs one itself; it always goes
through :func:`render_preview` to get a deterministic, agent-side-
trivial conversion from a :class:`PlanProposal`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.workflow.proposal import PlanProposal

__all__ = ["WorkflowPreviewView", "render_preview"]


class WorkflowPreviewView(BaseModel):
    """Derived view of a :class:`PlanProposal`.

    Three views, no IR:

    - ``python_script`` — a Python rendering of the workflow.
    - ``mermaid`` — a Mermaid flowchart rendering.
    - ``intervention_points`` — names of explicit user-input anchors.

    The IR (``workflow_ir``) is deliberately omitted. Agents that
    need the IR pull it off the underlying :class:`PlanProposal`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_script: str = ""
    mermaid: str = ""
    intervention_points: tuple[str, ...] = ()


def render_preview(plan: PlanProposal) -> WorkflowPreviewView:
    """Render the derived view for ``plan``.

    Pure function: same input → same output, no side effects.
    Part A.1 ships a minimal renderer (only the intervention-point
    names are populated); script + mermaid renderings will land in
    a follow-up spec when the workflow compiler grows them.
    """
    return WorkflowPreviewView(
        intervention_points=tuple(ip.name for ip in plan.intervention_points),
    )
