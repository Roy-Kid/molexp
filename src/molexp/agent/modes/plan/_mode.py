"""``PlanMode`` — the read-only typed planner driven on the harness.

After the ``plan-mode-pydanticai-rewrite`` collapse, PlanMode is a
**5-stage** :class:`~molexp.agent.mode.AgentMode`:

1. ``SynthesizeIntent`` — one structured call.
2. ``ClarifyIntent`` — pure routing.
3. ``ResearchAndPlan`` — one MCP-attached pydantic-ai agentic call that
   researches the toolchain via molmcp and emits a typed ``PlanGraph``
   with ``api_refs`` + ``composition_notes`` inline. Replaces the
   previous ``ExploreCapabilities`` + ``SynthesizeCandidates`` +
   ``SelectPlan`` triple.
4. ``PreflightPlanGraph`` — pure structural preflight (5 checks).
5. ``EmitApprovedPlan`` — ``approve_direction`` gate + handoff emission.

The substrate's :func:`execute_pipeline` walks the stages with two
:class:`RepairPolicy` rewinds — both rewinding to ``ResearchAndPlan``
on a preflight or rejection failure, bounded by
``config.max_repair_iterations``.

Plan artefacts persist through the bound :class:`PlanFolder`;
conversation entries go through the harness :class:`Session`.

The previous ``probe`` surface (``capability_probe=`` / ``probe_model=``
constructor kwargs, ``_resolve_probe``, ``PydanticAICapabilityProbe``)
is gone — the single ``ResearchAndPlan`` agent handles research +
planning in one MCP-attached call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.repair import RepairPolicy
from molexp.agent.harness.stage import NameOnlyStage, Stage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes._planning import (
    IllegalPlanTransitionError,
    PlanState,
)
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.plan.stages import (
    ClarifyIntent,
    EmitApprovedPlan,
    PreflightPlanGraph,
    ResearchAndPlan,
    SynthesizeIntent,
)
from molexp.agent.types import Message

# The research_planner is a pydantic-ai ``Agent[None, PlanGraph]`` —
# typed as ``Any`` here so this module stays inside the agent-layer
# firewall (no top-level ``import pydantic_ai``). The concrete agent
# instance is built behind the ``_pydanticai/`` boundary in
# ``_build_research_planner``.
__all__ = ["PlanMode", "PlanModeConfig"]


class PlanModeConfig(BaseModel):
    """Tunables for :class:`PlanMode`.

    Attributes:
        max_repair_iterations: Repair-loop budget for preflight-failure
            and rejected-direction repairs. ``0`` disables repair —
            the first failure is terminal.
    """

    model_config = ConfigDict(frozen=True)

    max_repair_iterations: int = 2


# Class-level declarative pipeline (NameOnlyStage placeholders) used by
# ``get_flowchart`` on bare instances + the no-drift guard. The real
# instance-level pipeline is built in ``__init__`` and shadows this
# attribute per instance.
_CLASS_PIPELINE = ModePipeline(
    stages=(
        NameOnlyStage("SynthesizeIntent"),
        NameOnlyStage("ClarifyIntent"),
        NameOnlyStage("ResearchAndPlan"),
        NameOnlyStage("PreflightPlanGraph"),
        NameOnlyStage("EmitApprovedPlan"),
    ),
    entry="SynthesizeIntent",
    edges=(
        PipelineEdge(from_stage="SynthesizeIntent", to_stage="ClarifyIntent"),
        # "ok" edge listed FIRST so the substrate's default routing picks
        # the happy path; the clarification branch is event-driven via
        # the ``clarification_required`` RepairPolicy.
        PipelineEdge(from_stage="ClarifyIntent", to_stage="ResearchAndPlan", label="ok"),
        PipelineEdge(
            from_stage="ClarifyIntent",
            to_stage="needs_clarification",
            label="blocked",
        ),
        PipelineEdge(from_stage="ResearchAndPlan", to_stage="PreflightPlanGraph"),
        PipelineEdge(
            from_stage="PreflightPlanGraph",
            to_stage="EmitApprovedPlan",
            label="pass",
        ),
        PipelineEdge(
            from_stage="PreflightPlanGraph",
            to_stage="ResearchAndPlan",
            label="fail, repair",
        ),
        PipelineEdge(
            from_stage="PreflightPlanGraph",
            to_stage="preflight_failed",
            label="fail, exhausted",
        ),
        PipelineEdge(from_stage="EmitApprovedPlan", to_stage="approved", label="approved"),
        PipelineEdge(
            from_stage="EmitApprovedPlan",
            to_stage="ResearchAndPlan",
            label="rejected, repair",
        ),
        PipelineEdge(
            from_stage="EmitApprovedPlan",
            to_stage="rejected",
            label="rejected, exhausted",
        ),
    ),
    terminal_states=(
        "approved",
        "needs_clarification",
        "preflight_failed",
        "rejected",
    ),
)


class PlanMode(AgentMode):
    """The read-only typed planner — five first-class Stages, no codegen."""

    name = "plan"
    pipeline = _CLASS_PIPELINE

    def __init__(
        self,
        *,
        config: PlanModeConfig | None = None,
        plan_folder: PlanFolder,
        research_planner: Any | None = None,  # noqa: ANN401 — pydantic-ai Agent kept behind firewall
        molmcp_command: str = "molmcp",
        molmcp_args: tuple[str, ...] = (),
        molmcp_env: dict[str, str] | None = None,
        planner_model: object | None = None,
        workspace: Path | None = None,
    ) -> None:
        """Construct the mode.

        Args:
            config: Tunables (repair budget). Defaults to
                :class:`PlanModeConfig`.
            plan_folder: The :class:`PlanFolder` artefacts persist
                through.
            research_planner: An explicitly-injected pydantic-ai
                ``Agent[None, PlanGraph]`` — used by tests. Wins over
                lazy construction from the model kwargs.
            molmcp_command: Executable for the molmcp MCP server (used
                when ``research_planner`` is None and ``planner_model``
                is set).
            molmcp_args: Optional CLI args for the MCP server.
            molmcp_env: Optional environment overlay for the MCP server.
            planner_model: pydantic-ai model identifier (e.g.
                ``"deepseek:deepseek-v4-flash"``). When set and
                ``research_planner`` is None, the agent is built lazily
                from this + the molmcp configuration on first
                ``ResearchAndPlan`` entry.
            workspace: Optional workspace root (unused by this layer;
                kept for symmetry with sibling modes).
        """
        self.config = config or PlanModeConfig()
        self.plan_folder = plan_folder
        self._injected_planner = research_planner
        self._planner_model = planner_model
        self._molmcp_command = molmcp_command
        self._molmcp_args = molmcp_args
        self._molmcp_env = molmcp_env
        self._workspace = workspace
        self._handoff: ApprovedPlanHandoff | None = None
        # Shadow the class-level declarative pipeline with the executable
        # one (carries real Stage instances + RepairPolicy + lifecycle).
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> ModePipeline:
        """Construct the executable :class:`ModePipeline` for this run."""
        max_iter = self.config.max_repair_iterations
        return ModePipeline(
            stages=(
                SynthesizeIntent(plan_mode=self),
                ClarifyIntent(plan_mode=self),
                ResearchAndPlan(plan_mode=self),
                PreflightPlanGraph(plan_mode=self),
                EmitApprovedPlan(plan_mode=self),
            ),
            entry="SynthesizeIntent",
            edges=_CLASS_PIPELINE.edges,
            terminal_states=_CLASS_PIPELINE.terminal_states,
            repairs=(
                # Clarification: max_iterations=0 means the first trigger
                # immediately routes to the terminal — used as an
                # event-driven "stop the pipeline at this terminal" gate.
                RepairPolicy(
                    trigger_event_kind="clarification_required",
                    rewind_to="needs_clarification",
                    max_iterations=0,
                    on_exhausted="needs_clarification",
                ),
                RepairPolicy(
                    trigger_event_kind="preflight_failed",
                    rewind_to="ResearchAndPlan",
                    max_iterations=max_iter,
                    on_exhausted="preflight_failed",
                ),
                RepairPolicy(
                    trigger_event_kind="repair_proposed",
                    rewind_to="ResearchAndPlan",
                    max_iterations=max_iter,
                    on_exhausted="rejected",
                ),
            ),
            lifecycle_validator=self._build_lifecycle_validator(),
        )

    def _build_lifecycle_validator(self) -> Callable[[Stage, AgentHarness], None]:
        """Return a callable that drives ``pre_state``-tagged transitions."""
        plan_mode = self

        def _validator(stage: Stage, _harness: AgentHarness) -> None:
            if stage.pre_state is None:
                return
            try:
                target = PlanState(stage.pre_state)
            except ValueError:
                return
            current = plan_mode.plan_folder.plan_state
            if current is target:
                return
            try:
                plan_mode.plan_folder.transition_to(target)
                plan_mode.plan_folder.save()
            except IllegalPlanTransitionError:
                pass

        return _validator

    def _build_research_planner(self) -> Any:  # noqa: ANN401 — pydantic-ai Agent
        """Return the ``ResearchAndPlan`` agent for this run.

        An explicitly-injected planner wins. Otherwise, when a model is
        configured, the production pydantic-ai-native agent is built
        lazily (behind the ``_pydanticai/`` firewall).
        """
        if self._injected_planner is not None:
            return self._injected_planner
        if self._planner_model is None:
            raise RuntimeError(
                "PlanMode.ResearchAndPlan needs either an injected research_planner "
                "or a planner_model="
            )
        from molexp.agent._pydanticai.research_planner import build_research_planner

        # The model identifier is opaque to molexp — pydantic-ai accepts
        # several shapes (model strings, KnownModelName, Model instances)
        # and the firewall keeps that union behind ``_pydanticai/``.
        return build_research_planner(
            cast("Any", self._planner_model),
            molmcp_command=self._molmcp_command,
            molmcp_args=self._molmcp_args,
            molmcp_env=self._molmcp_env,
        )

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the five-stage read-only pipeline, yielding events."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.session.append_message(Message(role="user", content=user_input))
        harness.router.clear_usage()
        self._handoff = None

        async for event in self.run_pipeline(
            harness=harness,
            user_input=user_input,
            initial_input=user_input,
        ):
            yield event

        # Post-pipeline finalize: on a rejection-exhausted run the
        # executor routes to ``rejected`` without transitioning
        # plan_folder. Complete the transition here.
        if self.plan_folder.plan_state is PlanState.awaiting_approval:
            self._transition(PlanState.rejected)

        yield self._build_completion(harness)

    def _build_completion(self, harness: AgentHarness) -> ModeCompletedEvent:
        """Fold the run into the terminal :class:`ModeCompletedEvent`."""
        breakdown = harness.router.snapshot_usage()
        terminal_state = self.plan_folder.plan_state
        mode_state: dict[str, object] = {"plan_state": terminal_state.value}
        if self._handoff is not None:
            mode_state["handoff"] = self._handoff.model_dump(mode="json")
            text = f"Plan {self._handoff.plan_id} approved."
        elif terminal_state is PlanState.needs_clarification:
            text = "Planning paused — clarification required."
        else:
            text = f"Planning ended in state {terminal_state.value}."
        harness.session.append_message(Message(role="assistant", content=text))
        result = AgentRunResult(
            text=text,
            messages=harness.session.build_context(),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        return ModeCompletedEvent(text=text, result=result.model_dump(mode="json"))

    def _transition(self, dst: PlanState) -> None:
        """Move the plan folder to ``dst`` (legal transitions only)."""
        if self.plan_folder.plan_state is dst:
            return
        self.plan_folder.transition_to(dst)
        self.plan_folder.save()
