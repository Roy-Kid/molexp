"""Stage 3 — research the toolchain via MCP, then emit a typed ``PlanGraph``.

This stage replaces the three earlier stages (``ExploreCapabilities`` +
``SynthesizeCandidates`` + ``SelectPlan``) with one pydantic-ai-native
agentic call. The agent — built in
:func:`molexp.agent._pydanticai.research_planner.build_research_planner` —
carries the molmcp toolset; pydantic-ai drives the model ↔ tool loop
internally. The LLM uses the catalog / search / lookup tools to study
what the toolchain offers, reasons about how to compose primitives, and
emits a typed :class:`PlanGraph` with ``api_refs`` + ``composition_notes``
inline on every :class:`PlanStep`.

On a repair rewind from ``preflight_failed`` (or ``repair_proposed`` from
the rejection gate), the stage prepends the prior plan + failure context
to the ``user`` prompt so the LLM regenerates with knowledge of what
failed. The pipeline's :class:`RepairPolicy` budget bounds how many times
this can happen.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar, cast

from molexp.agent.events import AgentEvent, ArtifactWrittenEvent
from molexp.agent.modes._planning import IntentSpec, PlanGraph
from molexp.agent.modes.plan.plan_graph_preflight import PlanGraphPreflightReport
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.stage import Stage

if TYPE_CHECKING:
    from molexp.agent.modes.plan._mode import PlanMode
    from molexp.agent.runtime import AgentHarness

__all__ = ["ResearchAndPlan"]


def _format_failure_context(
    prior_plan: PlanGraph | None,
    prior_preflight: PlanGraphPreflightReport | None,
) -> str:
    """Render the rewind context the agent sees when regenerating a plan.

    Empty on the first attempt. On a repair rewind, lists the failed
    preflight checks + the prior plan as JSON so the LLM knows exactly
    what to repair.
    """
    if prior_plan is None and prior_preflight is None:
        return ""
    blocks: list[str] = []
    if prior_preflight is not None:
        failed = tuple(check for check in prior_preflight.checks if not check.passed)
        if failed:
            lines = "\n".join(f"  - {check.name}: {check.detail}" for check in failed)
            blocks.append(f"PRIOR ATTEMPT FAILED these preflight checks:\n{lines}")
    if prior_plan is not None:
        blocks.append(
            "PRIOR ATTEMPT'S PlanGraph (regenerate, addressing the failures above):\n"
            f"{prior_plan.model_dump_json(indent=2)}"
        )
    return "\n\n".join(blocks)


def _build_user_prompt(intent: IntentSpec, rewind_context: str) -> str:
    """Assemble the ``agent.run(user=...)`` payload for ResearchAndPlan."""
    head = f"IntentSpec:\n{intent.model_dump_json(indent=2)}"
    if rewind_context:
        return f"{head}\n\n{rewind_context}"
    return head


class ResearchAndPlan(Stage[PlanThreadState, PlanThreadState]):
    """Drive one MCP-attached agentic call → typed :class:`PlanGraph`.

    Consumes the threaded :class:`PlanThreadState` (which carries the
    :class:`IntentSpec` and, on rewind, the prior plan + preflight
    report). Produces a new ``PlanThreadState`` with the ``plan_graph``
    field populated. Persists ``plan_graph.json`` to the plan folder and
    emits an :class:`ArtifactWrittenEvent`.

    The ``pre_state="exploring"`` tag drives the lifecycle validator:
    first entry transitions ``intake → exploring``; on a preflight-rewind
    it transitions ``preflight_failed → exploring``; on a rejection
    rewind ``awaiting_approval → exploring``. All three moves are legal
    per :data:`LEGAL_TRANSITIONS`.
    """

    name: ClassVar[str] = "ResearchAndPlan"
    pre_state: ClassVar[str | None] = "exploring"
    post_state: ClassVar[str | None] = "draft_plan"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — kept for substrate's Stage.run contract
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.intent is not None, "ResearchAndPlan ran without an intent"

        rewind_context = _format_failure_context(state.plan_graph, state.preflight)
        user_prompt = _build_user_prompt(state.intent, rewind_context)

        agent = self._plan_mode._build_research_planner()
        async with agent:
            result = await agent.run(user_prompt)
        plan_graph = cast(PlanGraph, result.output)

        path = self._plan_mode.plan_folder.write_plan_graph(plan_graph)
        yield ArtifactWrittenEvent(path=str(path), description="typed PlanGraph")
        # Clear preflight on regeneration so the next preflight stage runs fresh.
        yield state.model_copy(update={"plan_graph": plan_graph, "preflight": None})
