"""``RunMode`` — execute, monitor, and repair the materialized workflow.

RunMode is the fourth pipeline mode (Plan / Author / Run / Review). It
consumes AuthorMode's
:class:`~molexp.agent.modes.author.handoff.MaterializedWorkspaceHandoff`,
demands the third human gate
(:data:`~molexp.agent.modes._planning.ApprovalGate.approve_execution`),
loads the LLM-authored :class:`molexp.workflow.Workflow` through the
public ``molexp.workflow`` API, binds it to a workspace ``Run``, and
drives it to completion.

RunMode's own orchestration is a **plain async stage sequence** on the
harness — ``async with harness.stage(name): ...``. It *executes* the
generated workflow through the public ``molexp.workflow`` API (that
legitimately uses the workflow engine — RunMode is running the
materialized experiment). No ``pydantic_graph`` import.

Lifecycle: RunMode enters at :data:`PlanState.ready_for_run` and moves
``ready_for_run → running`` then ``running → completed`` (success),
``running → failed`` (unrecoverable structural failure), or
``running → needs_clarification`` (a repair needing AuthorMode re-entry).
Every move goes through :func:`assert_legal_transition` via
:meth:`PlanFolder.transition_to`.

On unrecoverable runtime failure RunMode classifies the failure, retries
transient ones per each step's ``RetryPolicy``, and on exhaustion emits a
structured :class:`~molexp.agent.modes._planning.PlanDiff`. A diff
needing re-materialization is wrapped in a
:class:`~molexp.agent.modes.run.repair.RepairEscalation` toward AuthorMode.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ErrorEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    RepairProposedEvent,
)
from molexp.agent.harness.stage import NameOnlyStage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes._planning import (
    IllegalPlanTransitionError,
    PlanGraph,
    PlanState,
)
from molexp.agent.modes.run.executor import ExecutionOutcome, RunExecutor
from molexp.agent.modes.run.gate import approve_execution_gate
from molexp.agent.modes.run.loader import WorkflowLoadError, load_materialized_workflow
from molexp.agent.modes.run.repair import (
    PlanDiff,
    RepairEscalation,
    RuntimeFailure,
    RuntimeFailureKind,
    build_repair_diff,
    build_repair_escalation,
)
from molexp.agent.modes.run.run_folder import RunFolder, RunReport
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff
    from molexp.agent.modes.plan.plan_folder import PlanFolder
    from molexp.workflow import Workflow
    from molexp.workspace import Experiment

_LOG = get_logger(__name__)

__all__ = ["RunMode", "RunModeConfig"]


class RunModeConfig(BaseModel):
    """Tunables for :class:`RunMode`.

    Attributes:
        retry_backoff_seconds: Pause between transient-failure retries.
            ``0.0`` (the default) makes retries immediate — used in tests.
        max_repair_escalations: Cap on :class:`RepairEscalation`\\ s a
            single run emits before giving up.
        require_execution_gate: When ``True`` (the default) the
            ``approve_execution`` gate must clear before anything is
            imported or executed; ``False`` bypasses it (test / trusted
            automation only).
    """

    model_config = ConfigDict(frozen=True)

    retry_backoff_seconds: float = 0.0
    max_repair_escalations: int = 1
    require_execution_gate: bool = True


class RunMode(AgentMode):
    """Execute, monitor, and repair an AuthorMode-materialized workflow."""

    name = "run"
    pipeline = ModePipeline(
        stages=(
            NameOnlyStage("LoadMaterializedWorkflow"),
            NameOnlyStage("ExecuteWorkflow"),
            NameOnlyStage("RepairRuntimeFailure"),
        ),
        entry="LoadMaterializedWorkflow",
        edges=(
            PipelineEdge(from_stage="LoadMaterializedWorkflow", to_stage="ExecuteWorkflow"),
            PipelineEdge(from_stage="ExecuteWorkflow", to_stage="completed", label="success"),
            PipelineEdge(
                from_stage="ExecuteWorkflow",
                to_stage="RepairRuntimeFailure",
                label="failure",
            ),
            PipelineEdge(
                from_stage="RepairRuntimeFailure",
                to_stage="ExecuteWorkflow",
                label="retry",
            ),
            PipelineEdge(
                from_stage="RepairRuntimeFailure",
                to_stage="repair_escalated",
                label="unrecoverable",
            ),
        ),
        terminal_states=("completed", "repair_escalated"),
    )

    def __init__(
        self,
        *,
        config: RunModeConfig | None = None,
        plan_folder: PlanFolder,
        experiment: Experiment,
        handoff: MaterializedWorkspaceHandoff | None = None,
    ) -> None:
        self.config = config or RunModeConfig()
        self.plan_folder = plan_folder
        self.experiment = experiment
        self._injected_handoff = handoff

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the execution pipeline, yielding orchestration events."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.session.append_message(Message(role="user", content=user_input))
        harness.router.clear_usage()

        handoff = self._injected_handoff
        if handoff is None:
            yield self._completion(
                harness,
                text="RunMode could not resolve a MaterializedWorkspaceHandoff.",
                terminal_state=self.plan_folder.plan_state,
            )
            return

        # Gate — nothing is imported or executed before it clears.
        gate_cleared = await self._clear_execution_gate(harness, handoff)
        if not gate_cleared:
            yield self._completion(
                harness,
                text=f"Execution of plan {handoff.plan_id} was not approved.",
                terminal_state=self.plan_folder.plan_state,
            )
            return

        async for event in self._run_after_gate(harness, handoff):
            yield event

    # ── gate ─────────────────────────────────────────────────────────────

    async def _clear_execution_gate(
        self, harness: AgentHarness, handoff: MaterializedWorkspaceHandoff
    ) -> bool:
        """Consult the ``approve_execution`` gate; return whether it cleared.

        With ``require_execution_gate=False`` the gate is bypassed
        entirely (no import / execution happens before this returns).
        """
        if not self.config.require_execution_gate:
            return True
        decision = await approve_execution_gate(handoff, harness=harness)
        return decision.approved

    # ── post-gate pipeline ───────────────────────────────────────────────

    async def _run_after_gate(
        self, harness: AgentHarness, handoff: MaterializedWorkspaceHandoff
    ) -> AsyncIterator[AgentEvent]:
        """Load, execute, monitor, and repair — the post-gate pipeline."""
        self._transition(PlanState.running)

        try:
            workflow = await self._load_stage(harness, handoff)
        except WorkflowLoadError as exc:
            await harness.emit(
                ErrorEvent(message=str(exc), error_type=type(exc).__name__, stage_name="load")
            )
            self._safe_transition(PlanState.failed)
            yield self._completion(
                harness,
                text=f"RunMode could not load the materialized workflow: {exc}",
                terminal_state=PlanState.failed,
            )
            return

        outcome = await self._execute_stage(harness, handoff, workflow)

        if outcome.succeeded:
            async for event in self._finish_success(harness, handoff, outcome):
                yield event
            return

        async for event in self._finish_failure(harness, handoff, outcome):
            yield event

    async def _load_stage(
        self, harness: AgentHarness, handoff: MaterializedWorkspaceHandoff
    ) -> Workflow:
        """Load the materialized workflow inside a harness stage."""
        async with harness.stage("LoadMaterializedWorkflow"):
            return load_materialized_workflow(handoff)

    async def _execute_stage(
        self,
        harness: AgentHarness,
        handoff: MaterializedWorkspaceHandoff,
        workflow: Workflow,
    ) -> ExecutionOutcome:
        """Bind the workflow to a workspace ``Run`` and execute it."""
        async with harness.stage("ExecuteWorkflow"):
            run = self.experiment.add_run(id=f"run-{handoff.plan_id}")
            executor = RunExecutor(
                workflow=workflow,
                run=run,
                plan_graph=handoff.plan_graph,
            )
            return await executor.execute()

    # ── success ──────────────────────────────────────────────────────────

    async def _finish_success(
        self,
        harness: AgentHarness,
        handoff: MaterializedWorkspaceHandoff,
        outcome: ExecutionOutcome,
    ) -> AsyncIterator[AgentEvent]:
        """Persist a completed run and yield the terminal event."""
        self._transition(PlanState.completed)
        report = RunReport(
            plan_id=handoff.plan_id,
            status="completed",
            run_id=outcome.run_id,
            execution_id=outcome.execution_id,
            progress=outcome.progress,
        )
        self._persist_report(report)
        yield self._completion(
            harness,
            text=f"Plan {handoff.plan_id} executed — {len(outcome.progress.steps)} step(s) completed.",
            terminal_state=PlanState.completed,
            report=report,
        )

    # ── failure + repair ─────────────────────────────────────────────────

    async def _finish_failure(
        self,
        harness: AgentHarness,
        handoff: MaterializedWorkspaceHandoff,
        outcome: ExecutionOutcome,
    ) -> AsyncIterator[AgentEvent]:
        """Classify the failure, emit a ``PlanDiff`` / escalation, persist."""
        async with harness.stage("RepairRuntimeFailure"):
            diff, escalation, failure = self._build_repair(handoff.plan_graph, outcome)

        if diff is not None:
            await harness.emit(
                RepairProposedEvent(
                    failed_invariant=diff.failed_invariant,
                    rationale=diff.rationale,
                )
            )

        terminal_state = (
            PlanState.needs_clarification
            if escalation is not None and escalation.requires_rematerialization
            else PlanState.failed
        )
        self._safe_transition(terminal_state)

        report = RunReport(
            plan_id=handoff.plan_id,
            status=terminal_state.value,
            run_id=outcome.run_id,
            execution_id=outcome.execution_id,
            progress=outcome.progress,
            repair_diffs=(diff,) if diff is not None else (),
            escalation=escalation,
        )
        self._persist_report(report)

        text = self._failure_text(handoff.plan_id, failure, terminal_state)
        yield self._completion(harness, text=text, terminal_state=terminal_state, report=report)

    def _build_repair(
        self, plan_graph: PlanGraph, outcome: ExecutionOutcome
    ) -> tuple[PlanDiff | None, RepairEscalation | None, RuntimeFailure | None]:
        """Build a :class:`PlanDiff` + :class:`RepairEscalation` for a failure.

        Identifies the failed step from the projected progress, classifies
        its failure structurally (a runtime failure that reached RunMode
        has already exhausted any per-step transient retries inside the
        executor), and builds the typed repair contract.
        """
        failed_ids = outcome.progress.failed_step_ids
        if not failed_ids:
            return None, None, None
        failed_id = failed_ids[0]
        failed_step = plan_graph.step_by_id(failed_id)
        if failed_step is None:
            return None, None, None

        progress_row = outcome.progress.step(failed_id)
        attempts = progress_row.attempts if progress_row is not None else 1
        failure = RuntimeFailure(
            step_id=failed_id,
            error_type=outcome.error_type or "WorkflowExecutionError",
            message=outcome.error_message or f"step {failed_id!r} failed at runtime",
            kind=RuntimeFailureKind.structural,
            attempts=max(attempts, 1),
        )
        diff = build_repair_diff(plan_graph=plan_graph, failed_step=failed_step, failure=failure)
        escalation = build_repair_escalation(plan_graph=plan_graph, diff=diff)
        return diff, escalation, failure

    # ── persistence ──────────────────────────────────────────────────────

    def _persist_report(self, report: RunReport) -> None:
        """Persist the :class:`RunReport` through a plan-anchored ``RunFolder``."""
        run_folder = self._run_folder(report.plan_id)
        run_folder.write_run_report(report)
        run_folder.write_progress("final", report.progress)
        if report.escalation is not None:
            run_folder.write_repair_escalation(report.escalation)

    def _run_folder(self, plan_id: str) -> RunFolder:
        """Mount (idempotently) the plan-anchored :class:`RunFolder`."""
        name = f"run-{plan_id}"
        if not self.plan_folder.has_folder(name, cls=RunFolder):
            self.plan_folder.add_folder(RunFolder(name=name, plan_id=plan_id))
        return self.plan_folder.get_folder(name, cls=RunFolder)

    # ── lifecycle ────────────────────────────────────────────────────────

    def _transition(self, dst: PlanState) -> None:
        """Move the plan folder to ``dst`` — legal transitions only."""
        if self.plan_folder.plan_state is dst:
            return
        self.plan_folder.transition_to(dst)
        self.plan_folder.save()

    def _safe_transition(self, dst: PlanState) -> None:
        """Best-effort transition to ``dst`` — swallow an illegal jump."""
        try:
            self._transition(dst)
        except IllegalPlanTransitionError:
            _LOG.warning(
                f"[run] cannot transition {self.plan_folder.plan_state.value} -> "
                f"{dst.value}; leaving plan state unchanged"
            )

    # ── terminal event ───────────────────────────────────────────────────

    @staticmethod
    def _failure_text(
        plan_id: str, failure: RuntimeFailure | None, terminal_state: PlanState
    ) -> str:
        """Render the terminal-event text for a failed run."""
        if terminal_state is PlanState.needs_clarification:
            return (
                f"Execution of plan {plan_id} needs re-materialization — escalated to AuthorMode."
            )
        if failure is not None:
            return (
                f"Execution of plan {plan_id} failed at step {failure.step_id!r}: "
                f"{failure.error_type}: {failure.message}"
            )
        return f"Execution of plan {plan_id} failed."

    def _completion(
        self,
        harness: AgentHarness,
        *,
        text: str,
        terminal_state: PlanState,
        report: RunReport | None = None,
    ) -> ModeCompletedEvent:
        """Fold the run into the terminal :class:`ModeCompletedEvent`."""
        breakdown = harness.router.snapshot_usage()
        mode_state: dict[str, object] = {"plan_state": terminal_state.value}
        if report is not None:
            mode_state["run"] = report.model_dump(mode="json")
        harness.session.append_message(Message(role="assistant", content=text))
        result = AgentRunResult(
            text=text,
            messages=harness.session.build_context(),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        return ModeCompletedEvent(text=text, result=result.model_dump(mode="json"))
