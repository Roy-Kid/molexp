"""``PlanMode`` — the read-only typed planner, driven on the harness.

PlanMode is an :class:`~molexp.agent.mode.AgentMode` with the
harness-based contract: ``run(*, harness, user_input)`` is an async
generator yielding :data:`~molexp.agent.harness.events.AgentEvent`\\ s.
It runs a **plain async sequence** of seven stages — it does *not*
build or run a ``molexp.workflow.Workflow``; it *produces* a typed
:class:`~molexp.agent.modes._planning.PlanGraph`.

Seven stages, each wrapped in ``harness.stage(...)``:

1. ``SynthesizeIntent`` — free text → typed ``IntentSpec``.
2. ``ClarifyIntent`` — blocking missing-info → ``needs_clarification``.
3. ``ExploreCapabilities`` — probe + project → typed ``CapabilityGraph``.
4. ``SynthesizeCandidates`` — one or three candidate ``PlanGraph``\\ s.
5. ``SelectPlan`` — choose one candidate.
6. ``PreflightPlanGraph`` — pure structural preflight.
7. ``EmitApprovedPlan`` — the ``approve_direction`` gate + handoff.

The repair loop (preflight failure / rejected direction) is a plain
``while`` bounded by ``PlanModeConfig.max_repair_iterations``: it builds
a :class:`PlanDiff`, emits ``repair_proposed``, and re-runs stages 4-6.

Plan artefacts persist through the bound :class:`PlanFolder`;
conversation entries go through the harness :class:`Session`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ArtifactWrittenEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.modes._planning import (
    ApprovalGate,
    CapabilityGraph,
    IntentSpec,
    PlanGraph,
    PlanState,
)
from molexp.agent.modes.plan.capability_probe_null import NullCapabilityProbe
from molexp.agent.modes.plan.capability_projection import capability_projection
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.plan.plan_graph_preflight import (
    PlanGraphPreflightReport,
    run_plan_graph_preflight,
)
from molexp.agent.modes.plan.protocols import CapabilityProbe
from molexp.agent.modes.plan.state import PlanRuntimeState, RepairSignal
from molexp.agent.modes.plan.tasks_planning import (
    CandidateSet,
    build_repair_diff,
    clarify_intent,
    select_plan,
    synthesize_candidates,
    synthesize_intent,
)
from molexp.agent.types import Message, utc_now

_LOG = get_logger(__name__)

__all__ = ["PlanMode", "PlanModeConfig"]


class PlanModeConfig(BaseModel):
    """Tunables for :class:`PlanMode`.

    Attributes:
        max_repair_iterations: Repair-loop budget. ``0`` disables
            repair — the first preflight failure / rejected direction
            is terminal.
    """

    model_config = ConfigDict(frozen=True)

    max_repair_iterations: int = 2


class _StageOutcome:
    """Mutable scratchpad carrying the seven stages' artefacts.

    Plain runtime container — the pipeline mutates it across stages and
    repair iterations.
    """

    def __init__(self) -> None:
        self.intent: IntentSpec | None = None
        self.capabilities: CapabilityGraph | None = None
        self.candidates: CandidateSet | None = None
        self.selected: PlanGraph | None = None
        self.preflight: PlanGraphPreflightReport | None = None
        self.handoff: ApprovedPlanHandoff | None = None
        self.terminal_state: PlanState = PlanState.intake


class PlanMode(AgentMode):
    """The read-only typed planner — seven stages, no codegen."""

    name = "plan"

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
        # Resolved at the start of each run(); NullCapabilityProbe until then.
        self._probe: CapabilityProbe = NullCapabilityProbe()

    def _resolve_probe(self) -> CapabilityProbe:
        """Return the capability probe to use for ``ExploreCapabilities``.

        An explicitly injected probe wins. Otherwise, when a model is
        configured, the production molmcp-backed
        :class:`PydanticAICapabilityProbe` is built lazily (behind the
        ``_pydanticai/`` firewall — so ``import molexp.agent`` stays
        SDK-free). When no molmcp server is configured, the fail-closed
        :class:`NullCapabilityProbe` is used.
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
        outcome = _StageOutcome()

        clarification_needed = await self._run_intake(harness, user_input, outcome)
        if not clarification_needed:
            async for event in self._run_planning(harness, outcome):
                yield event

        yield self._build_completion(harness, outcome)

    # ── stages 1-3 — intake ──────────────────────────────────────────────

    async def _run_intake(
        self,
        harness: AgentHarness,
        user_input: str,
        outcome: _StageOutcome,
    ) -> bool:
        """Run stages 1-3; return ``True`` when clarification stops the run."""
        async with harness.stage("SynthesizeIntent"):
            outcome.intent = await synthesize_intent(router=harness.router, user_input=user_input)
        intent_path = self.plan_folder.write_intent(outcome.intent)
        await harness.emit(
            ArtifactWrittenEvent(path=str(intent_path), description="typed IntentSpec")
        )

        async with harness.stage("ClarifyIntent"):
            next_state, blocking = clarify_intent(intent=outcome.intent)
        if next_state is PlanState.needs_clarification:
            self._transition(PlanState.needs_clarification)
            outcome.terminal_state = PlanState.needs_clarification
            questions = "; ".join(item.question for item in blocking)
            harness.session.append_message(
                Message(
                    role="assistant",
                    content=f"Need clarification before planning: {questions}",
                )
            )
            return True
        self._transition(PlanState.exploring)

        async with harness.stage("ExploreCapabilities"):
            probe_result = await self._probe.probe(intent=outcome.intent)
            outcome.capabilities = capability_projection(probe_result)
        cap_path = self.plan_folder.write_capability_graph(outcome.capabilities)
        await harness.emit(
            ArtifactWrittenEvent(path=str(cap_path), description="typed CapabilityGraph")
        )
        return False

    # ── stages 4-7 — planning + repair loop ──────────────────────────────

    async def _run_planning(
        self,
        harness: AgentHarness,
        outcome: _StageOutcome,
    ) -> AsyncIterator[AgentEvent]:
        """Run stages 4-7 with the plan-diff repair loop."""
        assert outcome.intent is not None
        assert outcome.capabilities is not None
        self._transition(PlanState.draft_plan)

        iteration = 0
        while True:
            await self._run_candidate_cycle(harness, outcome)
            assert outcome.selected is not None
            assert outcome.preflight is not None

            if not outcome.preflight.passed:
                async for event in self._handle_preflight_failure(harness, outcome, iteration):
                    yield event
                if self._runtime.repair_signal is None:
                    outcome.terminal_state = PlanState.preflight_failed
                    return
                iteration += 1
                continue

            # Stage 7 — approval gate.
            decision_approved, event = await self._run_approval(harness, outcome)
            if event is not None:
                yield event
            if decision_approved:
                outcome.terminal_state = PlanState.approved
                return
            if iteration >= self.config.max_repair_iterations:
                self._transition(PlanState.rejected)
                outcome.terminal_state = PlanState.rejected
                return
            yield self._propose_rejection_repair(outcome, iteration)
            iteration += 1

    async def _run_candidate_cycle(
        self,
        harness: AgentHarness,
        outcome: _StageOutcome,
    ) -> None:
        """Run stages 4-6 once (synthesize → select → preflight)."""
        assert outcome.intent is not None
        assert outcome.capabilities is not None

        async with harness.stage("SynthesizeCandidates"):
            outcome.candidates = await synthesize_candidates(
                router=harness.router,
                intent=outcome.intent,
                capabilities=outcome.capabilities,
            )
        for candidate in outcome.candidates.candidates:
            self.plan_folder.write_candidate(candidate.label, candidate.plan_graph)

        async with harness.stage("SelectPlan"):
            selected = await select_plan(
                router=harness.router,
                candidates=outcome.candidates,
                capabilities=outcome.capabilities,
            )
        selected = self._apply_pending_repair(selected)
        outcome.selected = selected
        sel_path = self.plan_folder.write_selected_plan(selected)
        await harness.emit(
            ArtifactWrittenEvent(path=str(sel_path), description="selected PlanGraph")
        )
        await harness.emit(
            PlanEmittedEvent(plan_id=selected.plan_id, step_count=len(selected.steps))
        )

        async with harness.stage("PreflightPlanGraph"):
            outcome.preflight = run_plan_graph_preflight(
                plan_graph=selected,
                intent=outcome.intent,
                capabilities=outcome.capabilities,
            )
        self.plan_folder.write_preflight_report(outcome.preflight)

    def _apply_pending_repair(self, plan: PlanGraph) -> PlanGraph:
        """Apply (and clear) any pending repair diff to ``plan``."""
        signal = self._runtime.consume()
        if signal is None:
            return plan
        return signal.plan_diff.apply(plan)

    # ── preflight-failure repair ─────────────────────────────────────────

    async def _handle_preflight_failure(
        self,
        harness: AgentHarness,
        outcome: _StageOutcome,
        iteration: int,
    ) -> AsyncIterator[AgentEvent]:
        """Emit ``preflight_failed`` and, within budget, plant a repair."""
        assert outcome.preflight is not None
        assert outcome.selected is not None
        failed = outcome.preflight.failed_check_names()
        await harness.emit(PreflightFailedEvent(failed_checks=failed))
        self._transition(PlanState.preflight_failed)

        if iteration >= self.config.max_repair_iterations:
            # Budget exhausted — the plan rests in preflight_failed, the
            # spec's terminal state for a structurally invalid plan.
            return
        diff = build_repair_diff(
            failed_invariant=failed[0] if failed else "preflight",
            plan_graph=outcome.selected,
            rationale=("preflight failed: " + ", ".join(failed) + "; re-synthesize candidates"),
        )
        self._runtime.plant(RepairSignal(plan_diff=diff))
        self._runtime.iteration = iteration + 1
        self._transition(PlanState.draft_plan)
        yield RepairProposedEvent(
            failed_invariant=diff.failed_invariant,
            rationale=diff.rationale,
        )

    # ── approval gate ────────────────────────────────────────────────────

    async def _run_approval(
        self,
        harness: AgentHarness,
        outcome: _StageOutcome,
    ) -> tuple[bool, AgentEvent | None]:
        """Run the ``approve_direction`` gate; return (approved, event)."""
        assert outcome.selected is not None
        assert outcome.intent is not None
        assert outcome.capabilities is not None
        self._transition(PlanState.awaiting_approval)
        view = _DirectionView(
            summary=(
                f"Plan {outcome.selected.plan_id}: "
                f"{len(outcome.selected.steps)} step(s) for "
                f"{outcome.intent.objective!r}"
            )
        )
        decision = await harness.approve(ApprovalGate.approve_direction, view)
        if decision.approved:
            self._transition(PlanState.approved)
            approved_plan = outcome.selected.model_copy(update={"state": PlanState.approved})
            outcome.selected = approved_plan
            self.plan_folder.write_selected_plan(approved_plan)
            outcome.handoff = ApprovedPlanHandoff(
                plan_id=self.plan_folder.plan_id,
                intent=outcome.intent,
                plan_graph=approved_plan,
                capability_graph=outcome.capabilities,
                plan_folder_path=Path(str(self.plan_folder.path())),
                direction_approved_at=utc_now(),
            )
            return True, None
        return False, None

    def _propose_rejection_repair(self, outcome: _StageOutcome, iteration: int) -> AgentEvent:
        """Plant a repair diff for a rejected direction; return the event."""
        assert outcome.selected is not None
        diff = build_repair_diff(
            failed_invariant="direction_rejected",
            plan_graph=outcome.selected,
            rationale="the reviewer rejected the direction; re-synthesize candidates",
        )
        self._runtime.plant(RepairSignal(plan_diff=diff))
        self._runtime.iteration = iteration + 1
        self._transition(PlanState.draft_plan)
        return RepairProposedEvent(
            failed_invariant=diff.failed_invariant,
            rationale=diff.rationale,
        )

    # ── completion ───────────────────────────────────────────────────────

    def _build_completion(
        self, harness: AgentHarness, outcome: _StageOutcome
    ) -> ModeCompletedEvent:
        """Fold the run into the terminal :class:`ModeCompletedEvent`."""
        breakdown = harness.router.snapshot_usage()
        mode_state: dict[str, object] = {"plan_state": outcome.terminal_state.value}
        if outcome.handoff is not None:
            mode_state["handoff"] = outcome.handoff.model_dump(mode="json")
            text = f"Plan {outcome.handoff.plan_id} approved."
        elif outcome.terminal_state is PlanState.needs_clarification:
            text = "Planning paused — clarification required."
        else:
            text = f"Planning ended in state {outcome.terminal_state.value}."
        if outcome.preflight is not None:
            mode_state["preflight_passed"] = outcome.preflight.passed
        harness.session.append_message(Message(role="assistant", content=text))
        result = AgentRunResult(
            text=text,
            messages=harness.session.build_context(),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        return ModeCompletedEvent(text=text, result=result.model_dump(mode="json"))

    # ── lifecycle helpers ────────────────────────────────────────────────

    def _transition(self, dst: PlanState) -> None:
        """Move the plan folder to ``dst`` (legal transitions only)."""
        if self.plan_folder.plan_state is dst:
            return
        self.plan_folder.transition_to(dst)
        self.plan_folder.save()


class _DirectionView:
    """Minimal approval-view for the ``approve_direction`` gate.

    The harness's :meth:`AgentHarness.approve` only reads ``.summary``;
    this plain object satisfies that contract without a pydantic model.
    """

    def __init__(self, *, summary: str) -> None:
        self.summary = summary
