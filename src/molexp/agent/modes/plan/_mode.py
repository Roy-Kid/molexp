"""``PlanMode`` — the read-only typed planner driven on the harness.

PlanMode is an :class:`~molexp.agent.mode.AgentMode` with the
harness-based contract: ``run(*, harness, user_input)`` is an async
generator yielding :data:`~molexp.agent.harness.events.AgentEvent`\\ s.
After ``agent-mode-stage-pipeline-02``, the seven stages are
**first-class** :class:`~molexp.agent.harness.stage.Stage` subclasses
under :mod:`molexp.agent.modes.plan.stages`; the substrate's
:func:`~molexp.agent.harness.pipeline.execute_pipeline` walks them with
three registered :class:`~molexp.agent.harness.repair.RepairPolicy`
rewinds (clarification-required / preflight-failed / repair-proposed)
+ a :class:`PlanState` ``lifecycle_validator``.

PlanMode's own ``run`` body reduces to:

- emit :class:`ModeStartedEvent` and append the user turn;
- resolve the capability probe;
- delegate to :meth:`run_pipeline` (the substrate drains the seven
  stages);
- finalize ``awaiting_approval → rejected`` when a rejection
  exhausted the repair budget — the executor routes to the
  ``rejected`` terminal without transitioning ``plan_folder`` itself
  (the per-stage rule is that the rejection stage doesn't transition
  on its own, so the lifecycle stays consistent across rewinds);
- yield :class:`ModeCompletedEvent`.

Plan artefacts persist through the bound :class:`PlanFolder`;
conversation entries go through the harness :class:`Session`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

from mollog import get_logger
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
from molexp.agent.modes.plan.capability_probe_null import NullCapabilityProbe
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.plan.protocols import CapabilityProbe
from molexp.agent.modes.plan.stages import (
    ClarifyIntent,
    EmitApprovedPlan,
    ExploreCapabilities,
    PreflightPlanGraph,
    SelectPlan,
    SynthesizeCandidates,
    SynthesizeIntent,
)
from molexp.agent.modes.plan.state import PlanRuntimeState
from molexp.agent.types import Message

_LOG = get_logger(__name__)

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
# instance-level pipeline (with executable Stage instances + repair
# policies + lifecycle validator) is built in ``__init__`` and shadows
# this attribute on each instance.
_CLASS_PIPELINE = ModePipeline(
    stages=(
        NameOnlyStage("SynthesizeIntent"),
        NameOnlyStage("ClarifyIntent"),
        NameOnlyStage("ExploreCapabilities"),
        NameOnlyStage("SynthesizeCandidates"),
        NameOnlyStage("SelectPlan"),
        NameOnlyStage("PreflightPlanGraph"),
        NameOnlyStage("EmitApprovedPlan"),
    ),
    entry="SynthesizeIntent",
    edges=(
        PipelineEdge(from_stage="SynthesizeIntent", to_stage="ClarifyIntent"),
        # "ok" edge is FIRST so the substrate's default-routing picks
        # the happy path; clarification routing is event-driven via the
        # ``clarification_required`` RepairPolicy.
        PipelineEdge(from_stage="ClarifyIntent", to_stage="ExploreCapabilities", label="ok"),
        PipelineEdge(
            from_stage="ClarifyIntent",
            to_stage="needs_clarification",
            label="blocked",
        ),
        PipelineEdge(from_stage="ExploreCapabilities", to_stage="SynthesizeCandidates"),
        PipelineEdge(from_stage="SynthesizeCandidates", to_stage="SelectPlan"),
        PipelineEdge(from_stage="SelectPlan", to_stage="PreflightPlanGraph"),
        PipelineEdge(
            from_stage="PreflightPlanGraph",
            to_stage="EmitApprovedPlan",
            label="pass",
        ),
        PipelineEdge(
            from_stage="PreflightPlanGraph",
            to_stage="SynthesizeCandidates",
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
            to_stage="SynthesizeCandidates",
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
    """The read-only typed planner — seven first-class Stages, no codegen."""

    name = "plan"
    pipeline = _CLASS_PIPELINE

    def __init__(
        self,
        *,
        config: PlanModeConfig | None = None,
        plan_folder: PlanFolder,
        capability_probe: CapabilityProbe | None = None,
        probe_model: object | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.config = config or PlanModeConfig()
        self.plan_folder = plan_folder
        self._injected_probe = capability_probe
        self._probe_model = probe_model
        self._workspace = workspace
        self._runtime = PlanRuntimeState()
        self._probe: CapabilityProbe = NullCapabilityProbe()
        self._handoff: ApprovedPlanHandoff | None = None
        # Shadow the class-level declarative pipeline with the executable
        # one (carries real Stage instances + RepairPolicy + lifecycle).
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> ModePipeline:
        """Construct the executable :class:`ModePipeline` for this run.

        Real Stage instances replace the class-level ``NameOnlyStage``
        placeholders; three :class:`RepairPolicy` rewinds and the
        per-mode :class:`PlanState` lifecycle validator plug in.
        """
        max_iter = self.config.max_repair_iterations
        return ModePipeline(
            stages=(
                SynthesizeIntent(plan_mode=self),
                ClarifyIntent(plan_mode=self),
                ExploreCapabilities(plan_mode=self),
                SynthesizeCandidates(plan_mode=self),
                SelectPlan(plan_mode=self),
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
                    rewind_to="SynthesizeCandidates",
                    max_iterations=max_iter,
                    on_exhausted="preflight_failed",
                ),
                RepairPolicy(
                    trigger_event_kind="repair_proposed",
                    rewind_to="SynthesizeCandidates",
                    max_iterations=max_iter,
                    on_exhausted="rejected",
                ),
            ),
            lifecycle_validator=self._build_lifecycle_validator(),
        )

    def _build_lifecycle_validator(self) -> Callable[[Stage, AgentHarness], None]:
        """Return a callable that drives ``pre_state``-tagged transitions.

        The substrate calls it once per stage entry. The validator
        translates the stage's opaque ``pre_state`` string tag into a
        :class:`PlanState` value and transitions ``plan_folder`` when
        the current state differs from the target. Illegal moves are
        silently ignored — the executor's repair policies will route
        the pipeline to the correct terminal next.
        """
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

    def _resolve_probe(self) -> CapabilityProbe:
        """Return the capability probe to use for ``ExploreCapabilities``.

        An explicitly injected probe wins. Otherwise, when a model is
        configured, the production molmcp-backed
        :class:`PydanticAICapabilityProbe` is built lazily (behind the
        ``_pydanticai/`` firewall). When no molmcp server is
        configured, the fail-closed :class:`NullCapabilityProbe` is
        used.
        """
        if self._injected_probe is not None:
            return self._injected_probe
        if self._probe_model is not None:
            from molexp.agent._pydanticai.capability_probe_factory import (
                build_capability_probe,
            )

            probe = build_capability_probe(workspace=self._workspace, model=self._probe_model)
            if probe is not None:
                return probe
        return NullCapabilityProbe()

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the seven-stage read-only pipeline, yielding events."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.session.append_message(Message(role="user", content=user_input))
        harness.router.clear_usage()
        self._probe = self._resolve_probe()
        self._handoff = None
        self._runtime = PlanRuntimeState()

        async for event in self.run_pipeline(
            harness=harness,
            user_input=user_input,
            initial_input=user_input,
        ):
            yield event

        # Post-pipeline finalize: on a rejection-exhausted run the
        # executor routes to the ``rejected`` terminal without
        # transitioning plan_folder (EmitApprovedPlan deliberately
        # doesn't transition on reject — it lets the lifecycle
        # validator transition ``awaiting_approval → draft_plan`` on
        # rewind). Complete the transition to ``rejected`` here.
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
