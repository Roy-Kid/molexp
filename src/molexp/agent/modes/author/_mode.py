"""``AuthorMode`` — materialize an approved typed plan into a workspace.

AuthorMode is an :class:`~molexp.agent.mode.AgentMode` driven on the
harness. It consumes the :class:`~molexp.agent.modes.plan.handoff.ApprovedPlanHandoff`
PlanMode emits, lowers the typed :class:`~molexp.agent.modes._planning.PlanGraph`
into a :class:`~molexp.workflow.WorkflowContract`, generates the
experiment workspace (workflow IR, per-task source, per-task tests, the
package skeleton, a manifest), runs each generated task's test through an
isolated-subprocess debug loop, and emits a
:class:`~molexp.agent.modes.author.handoff.MaterializedWorkspaceHandoff`.

AuthorMode's own pipeline is a **plain async stage sequence** on the
harness — ``async with harness.stage(name): ...`` — not a
``pydantic_graph`` graph. (It *produces* a runnable
:class:`~molexp.workflow.Workflow`, which legitimately uses the public
``molexp.workflow`` API.)

The ``approve_materialization`` gate runs once, before any file is
written; a rejected gate writes no source and moves the plan to
:data:`~molexp.agent.modes._planning.PlanState.failed`. The plan moves
``approved → materializing → validating → ready_for_run`` (or
``failed``) via :func:`~molexp.agent.modes._planning.assert_legal_transition`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from pathlib import Path

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ArtifactWrittenEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    RepairProposedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes._planning import (
    ApprovalGate,
    PlanGraph,
    PlanState,
)
from molexp.agent.modes.author.codegen import GeneratedModule
from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff
from molexp.agent.modes.author.materialize import (
    MaterializationOutcome,
    materialize_plan,
)
from molexp.agent.modes.author.workspace_layout import MaterializedLayout
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.router import ModelTier
from molexp.agent.types import Message

_LOG = get_logger(__name__)

__all__ = ["AuthorMode", "AuthorModeConfig"]


class AuthorModeConfig(BaseModel):
    """Tunables for :class:`AuthorMode`.

    Attributes:
        debug_attempts: Per-task debug-loop run budget. ``1`` runs each
            generated test once with no repair; ``>=2`` enables the
            run→repair→re-run loop.
        subprocess_timeout_seconds: Hard timeout for each isolated
            pytest subprocess.
        codegen_tier: Model tier the per-task codegen / repair LLM calls
            run at.
    """

    model_config = ConfigDict(frozen=True)

    debug_attempts: int = 2
    subprocess_timeout_seconds: float = 30.0
    codegen_tier: ModelTier = ModelTier.DEFAULT


class _MaterializationView:
    """Minimal approval-view for the ``approve_materialization`` gate.

    :meth:`AgentHarness.approve` reads only ``.summary``; this plain
    object satisfies that contract without a pydantic model.
    """

    def __init__(self, *, summary: str) -> None:
        self.summary = summary


class AuthorMode(AgentMode):
    """Materialize an approved typed plan into a validated experiment workspace."""

    name = "author"
    pipeline = ModePipeline(
        stages=(
            "LowerPlanGraph",
            "CompileTaskIR",
            "GenerateWorkflowSkeleton",
            "GenerateTaskTests",
            "GenerateTaskImplementations",
            "RunTaskDebugLoop",
            "ValidateWorkspace",
            "WriteManifest",
        ),
        edges=(
            PipelineEdge(from_stage="LowerPlanGraph", to_stage="CompileTaskIR"),
            PipelineEdge(from_stage="CompileTaskIR", to_stage="GenerateWorkflowSkeleton"),
            PipelineEdge(from_stage="GenerateWorkflowSkeleton", to_stage="GenerateTaskTests"),
            PipelineEdge(
                from_stage="GenerateTaskTests",
                to_stage="GenerateTaskImplementations",
            ),
            PipelineEdge(from_stage="GenerateTaskImplementations", to_stage="RunTaskDebugLoop"),
            PipelineEdge(from_stage="RunTaskDebugLoop", to_stage="ValidateWorkspace"),
            PipelineEdge(from_stage="ValidateWorkspace", to_stage="WriteManifest"),
            PipelineEdge(from_stage="WriteManifest", to_stage="materialized"),
        ),
        terminal_states=("materialized",),
    )

    def __init__(
        self,
        *,
        config: AuthorModeConfig | None = None,
        plan_folder: PlanFolder,
        handoff: ApprovedPlanHandoff | None = None,
        repair_model: object | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.config = config or AuthorModeConfig()
        self.plan_folder = plan_folder
        self._injected_handoff = handoff
        # When ``repair_model`` is set, the RunTaskDebugLoop stage routes
        # the LLM repair through the source-grounded MCP-attached agent
        # built behind the ``_pydanticai/`` firewall; otherwise the loop
        # falls back to its legacy no-tool router path.
        self._repair_model = repair_model
        self._workspace = workspace

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the materialization pipeline, yielding orchestration events."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.session.append_message(Message(role="user", content=user_input))
        harness.router.clear_usage()

        handoff = self._resolve_handoff(harness, user_input)
        if handoff is None:
            yield self._completion(
                harness,
                text="AuthorMode could not resolve an ApprovedPlanHandoff.",
                terminal_state=PlanState.failed,
            )
            return

        # Gate before any file is written.
        decision = await harness.approve(
            ApprovalGate.approve_materialization,
            _MaterializationView(
                summary=(
                    f"Materialize plan {handoff.plan_id}: {len(handoff.plan_graph.steps)} step(s)"
                )
            ),
        )
        if not decision.approved:
            # The lifecycle reaches `failed` only via `materializing`;
            # a rejected gate steps through it without writing any file.
            self._transition(PlanState.materializing, source=PlanState.approved)
            self._transition(PlanState.failed, source=PlanState.materializing)
            yield self._completion(
                harness,
                text=f"Materialization of plan {handoff.plan_id} was rejected.",
                terminal_state=PlanState.failed,
            )
            return

        result_handoff: MaterializedWorkspaceHandoff | None = None
        terminal_state = PlanState.failed
        try:
            outcome = await self._run_pipeline(harness, handoff, decision_at=decision)
            terminal_state = outcome.terminal_state
            result_handoff = outcome.handoff
            for event in outcome.events:
                yield event
        except Exception as exc:
            _LOG.error(f"[author] pipeline failed: {type(exc).__name__}: {exc}")
            self._safe_transition(PlanState.failed)
            yield self._completion(
                harness,
                text=f"Materialization failed: {exc}",
                terminal_state=PlanState.failed,
            )
            return

        yield self._completion(
            harness,
            text=self._terminal_text(handoff.plan_id, terminal_state),
            terminal_state=terminal_state,
            handoff=result_handoff,
        )

    # ── pipeline ─────────────────────────────────────────────────────────

    def _build_repair_callable(
        self,
    ) -> Callable[[str], Awaitable[GeneratedModule]] | None:
        """Lazily build the MCP-attached repair callable for the debug loop.

        Returns ``None`` when no repair model was configured, in which
        case the debug loop falls back to its legacy no-tool router
        path. The ``_pydanticai.debug_repair`` import is deferred so
        ``import molexp.agent.modes.author`` stays SDK-free.
        """
        if self._repair_model is None:
            return None
        from molexp.agent._pydanticai.debug_repair import build_repair_callable

        return build_repair_callable(
            workspace=self._workspace,
            model=self._repair_model,
        )

    async def _run_pipeline(
        self,
        harness: AgentHarness,
        handoff: ApprovedPlanHandoff,
        *,
        decision_at: object,
    ) -> _PipelineOutcome:
        """Run the codegen + debug + validation stages; return the outcome."""
        self._transition(PlanState.materializing, source=PlanState.approved)
        repair = self._build_repair_callable()
        outcome: MaterializationOutcome = await materialize_plan(
            harness=harness,
            handoff=handoff,
            layout=self._layout(),
            config=self.config,
            repair=repair,
        )
        for path in outcome.artifact_paths:
            await harness.emit(ArtifactWrittenEvent(path=path, description="materialized artefact"))

        events: list[AgentEvent] = []
        if not outcome.codegen_ok:
            for diff in outcome.repair_diffs:
                events.append(
                    RepairProposedEvent(
                        failed_invariant=diff.failed_invariant,
                        rationale=diff.rationale,
                    )
                )
            self._transition(PlanState.failed, source=PlanState.materializing)
            return _PipelineOutcome(
                terminal_state=PlanState.failed, handoff=None, events=tuple(events)
            )

        self._transition(PlanState.validating, source=PlanState.materializing)
        if not outcome.validation_report.ok:
            self._transition(PlanState.failed, source=PlanState.validating)
            return _PipelineOutcome(
                terminal_state=PlanState.failed, handoff=None, events=tuple(events)
            )

        self._transition(PlanState.ready_for_run, source=PlanState.validating)
        ready_plan = self._mark_ready(outcome.plan_graph)
        materialized = MaterializedWorkspaceHandoff(
            plan_id=handoff.plan_id,
            plan_graph=ready_plan,
            experiment_workspace_path=outcome.experiment_workspace_path,
            workflow_yaml_path=outcome.workflow_yaml_path,
            entrypoint_module=outcome.entrypoint_module,
            entrypoint_symbol=outcome.entrypoint_symbol,
            source_root=outcome.source_root,
            validation_report_snapshot=outcome.validation_report,
            materialization_approved_at=_decision_time(decision_at),
        )
        return _PipelineOutcome(
            terminal_state=PlanState.ready_for_run,
            handoff=materialized,
            events=tuple(events),
        )

    # ── handoff resolution ───────────────────────────────────────────────

    def _resolve_handoff(
        self, harness: AgentHarness, user_input: str
    ) -> ApprovedPlanHandoff | None:
        """Resolve the :class:`ApprovedPlanHandoff` to materialize.

        The handoff is supplied at construction by the orchestrator that
        chains PlanMode → AuthorMode (PlanMode emits the handoff in its
        terminal :class:`ModeCompletedEvent`; the orchestrator validates
        it and constructs :class:`AuthorMode` with it). ``user_input``
        is advisory only. Returns ``None`` when no handoff was supplied —
        AuthorMode then ends in :data:`PlanState.failed`.
        """
        del harness, user_input  # the handoff is injected at construction
        return self._injected_handoff

    # ── helpers ──────────────────────────────────────────────────────────

    def _layout(self) -> MaterializedLayout:
        """Return the :class:`MaterializedLayout` over the bound plan folder."""
        return MaterializedLayout(self.plan_folder)

    @staticmethod
    def _mark_ready(plan_graph: PlanGraph) -> PlanGraph:
        """Return ``plan_graph`` with state set to ``ready_for_run``."""
        return plan_graph.model_copy(update={"state": PlanState.ready_for_run})

    @staticmethod
    def _terminal_text(plan_id: str, state: PlanState) -> str:
        """Render the terminal-event text for a finished run."""
        if state is PlanState.ready_for_run:
            return f"Plan {plan_id} materialized — workspace ready_for_run."
        return f"Materialization of plan {plan_id} ended in state {state.value}."

    def _transition(self, dst: PlanState, *, source: PlanState) -> None:
        """Move the plan folder to ``dst`` (legal transitions only).

        ``source`` documents the expected origin state; the
        ``PlanFolder`` enforces the legal-transition table.
        """
        del source  # documented expectation; PlanFolder enforces legality
        if self.plan_folder.plan_state is dst:
            return
        self.plan_folder.transition_to(dst)
        self.plan_folder.save()

    def _safe_transition(self, dst: PlanState) -> None:
        """Best-effort transition to ``dst`` — swallow an illegal jump."""
        from molexp.agent.modes._planning import IllegalPlanTransitionError

        try:
            self._transition(dst, source=self.plan_folder.plan_state)
        except IllegalPlanTransitionError:
            _LOG.warning(
                f"[author] cannot transition {self.plan_folder.plan_state.value} -> "
                f"{dst.value}; leaving plan state unchanged"
            )

    def _completion(
        self,
        harness: AgentHarness,
        *,
        text: str,
        terminal_state: PlanState,
        handoff: MaterializedWorkspaceHandoff | None = None,
    ) -> ModeCompletedEvent:
        """Fold the run into the terminal :class:`ModeCompletedEvent`."""
        breakdown = harness.router.snapshot_usage()
        mode_state: dict[str, object] = {"plan_state": terminal_state.value}
        if handoff is not None:
            mode_state["handoff"] = handoff.model_dump(mode="json")
        harness.session.append_message(Message(role="assistant", content=text))
        result = AgentRunResult(
            text=text,
            messages=harness.session.build_context(),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        return ModeCompletedEvent(text=text, result=result.model_dump(mode="json"))


class _PipelineOutcome:
    """Plain runtime container for one pipeline run's result."""

    def __init__(
        self,
        *,
        terminal_state: PlanState,
        handoff: MaterializedWorkspaceHandoff | None,
        events: tuple[AgentEvent, ...],
    ) -> None:
        self.terminal_state = terminal_state
        self.handoff = handoff
        self.events = events


def _decision_time(decision_at: object) -> datetime:
    """Extract a timestamp from an approval decision, falling back to now."""
    from molexp.agent.types import utc_now

    if isinstance(decision_at, ApprovalDecidedEvent):
        return decision_at.timestamp
    return utc_now()
