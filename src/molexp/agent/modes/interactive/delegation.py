"""PlanMode delegation for the emergent :class:`InteractiveMode`.

InteractiveMode is read-only — it never writes executable code. The
auditable output path (refine → decompose → review → codegen) stays
owned by the structured :class:`~molexp.agent.modes.plan.PlanMode`.
This module is the single seam between the two:

- :func:`delegate_to_plan` drives PlanMode end-to-end on the *same*
  :class:`~molexp.agent.harness.harness.AgentHarness` InteractiveMode is
  running, so PlanMode's own :data:`AgentEvent` stream surfaces in the
  parent event flow, and returns a one-line handoff summary.
- :func:`run_plan_pipeline_tool` wraps that call as a pydantic-ai tool
  the emergent LLM loop can dispatch autonomously.

Both the human ``/plan`` slash command and the LLM-dispatched
``run_plan_pipeline`` tool funnel through :func:`delegate_to_plan`.
PlanMode itself is unchanged — InteractiveMode *composes* it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

from mollog import get_logger

from molexp.agent.harness.events import ModeCompletedEvent
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.modes.plan import PlanFolder, PlanMode
from molexp.workspace import Workspace

_LOG = get_logger(__name__)

__all__ = ["delegate_to_plan", "run_plan_pipeline_tool"]


def _summarize_handoff(plan_id: str, terminal: ModeCompletedEvent | None) -> str:
    """Render the PlanMode run's terminal event into a one-line summary."""
    if terminal is None:
        return f"PlanMode delegation for plan {plan_id!r} produced no completion event."
    plan_state = ""
    if terminal.result:
        mode_state = terminal.result.get("mode_state") or {}
        plan_state = str(mode_state.get("plan_state", ""))
    base = terminal.text or f"PlanMode finished for plan {plan_id!r}."
    if plan_state:
        return f"{base} (plan={plan_id}, state={plan_state})"
    return f"{base} (plan={plan_id})"


async def delegate_to_plan(
    harness: AgentHarness,
    preliminary_plan: str,
    *,
    workspace_root: Path,
) -> str:
    """Run the structured PlanMode pipeline on ``harness`` and summarize it.

    Mounts a fresh :class:`PlanFolder` on the workspace, drives
    :meth:`PlanMode.run` on the *same* harness InteractiveMode is using,
    and re-surfaces every event PlanMode *yields* into that harness's
    sink (the events PlanMode ``emit``\\ s already land there directly).
    The whole PlanMode :data:`AgentEvent` stream therefore appears in the
    parent flow.

    Args:
        harness: The live harness InteractiveMode is driving.
        preliminary_plan: The user's preliminary experiment plan.
        workspace_root: Workspace the plan artefacts persist under.

    Returns:
        A non-empty one-line summary of PlanMode's terminal handoff.
    """
    Path(workspace_root).mkdir(parents=True, exist_ok=True)
    workspace = Workspace(workspace_root)
    plan_folder = cast("PlanFolder", workspace.add_folder(PlanFolder()))
    plan_mode = PlanMode(plan_folder=plan_folder)
    _LOG.info(f"[interactive] delegating to PlanMode (plan={plan_folder.plan_id})")

    terminal: ModeCompletedEvent | None = None
    async for event in plan_mode.run(harness=harness, user_input=preliminary_plan):
        # PlanMode emits most events straight to the shared sink; the few
        # it *yields* (repair proposals, its terminal completion) are
        # re-surfaced here so the parent stream is complete.
        await harness.emit(event)
        if isinstance(event, ModeCompletedEvent):
            terminal = event
    return _summarize_handoff(plan_folder.plan_id, terminal)


def run_plan_pipeline_tool(
    harness: AgentHarness,
    workspace_root: Path,
) -> Callable[[str], Awaitable[str]]:
    """Build the ``run_plan_pipeline`` pydantic-ai tool for the loop.

    A closure factory: the returned async callable closes over the
    harness + workspace so the emergent LLM loop sees only a single
    ``preliminary_plan`` argument. Its docstring is what pydantic-ai
    shows the model as the tool description.

    Args:
        harness: The live harness the InteractiveMode loop drives.
        workspace_root: Workspace PlanMode persists plan artefacts under.

    Returns:
        The ``run_plan_pipeline`` async tool callable.
    """

    async def run_plan_pipeline(preliminary_plan: str) -> str:
        """Hand a preliminary experiment plan to the structured planning pipeline.

        Use this when the user wants a concrete, reviewable experiment
        plan or workflow. The planning pipeline refines the plan,
        decomposes it into testable steps, runs a structural preflight,
        and takes it through a human approval gate — producing an
        auditable plan rather than a free-form answer.

        Args:
            preliminary_plan: The user's preliminary experiment plan, in
                free text.

        Returns:
            A one-line summary of the planning pipeline's outcome.
        """
        return await delegate_to_plan(harness, preliminary_plan, workspace_root=workspace_root)

    return run_plan_pipeline
