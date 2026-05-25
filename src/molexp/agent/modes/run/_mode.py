"""``RunMode`` — execute, monitor, and repair the materialized workflow.

RunMode is the fourth pipeline mode (Plan / Author / Run / Review). It
consumes AuthorMode's
:class:`~molexp.agent.modes.author.handoff.MaterializedWorkspaceHandoff`,
demands the third human gate
(:data:`~molexp.agent.modes._planning.ApprovalGate.approve_execution`),
loads the LLM-authored :class:`molexp.workflow.Workflow` through the
public ``molexp.workflow`` API, binds it to a workspace ``Run``, and
drives it to completion.

After ``agent-mode-stage-pipeline-03``, the three stages live as
first-class :class:`~molexp.agent.modes.run.stages.LoadMaterializedWorkflow`
/ :class:`ExecuteWorkflow` / :class:`RepairRuntimeFailure` Stage
subclasses; :meth:`RunMode.run` delegates to
:meth:`AgentMode.run_pipeline`. Pre-pipeline ``run`` still handles
the handoff-missing and gate-blocked early returns inline — there is
no point importing or executing anything before they clear.

Lifecycle: RunMode enters at :data:`PlanState.ready_for_run` and moves
``ready_for_run → running`` then ``running → completed`` (success),
``running → failed`` (unrecoverable structural failure), or
``running → needs_clarification`` (a repair needing AuthorMode re-entry).
Every move goes through :func:`assert_legal_transition` via
:meth:`PlanFolder.transition_to`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
)
from molexp.agent.harness.stage import NameOnlyStage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes._planning import IllegalPlanTransitionError, PlanState
from molexp.agent.modes.run.executor import ExecutionOutcome
from molexp.agent.modes.run.gate import approve_execution_gate
from molexp.agent.modes.run.loader import WorkflowLoadError, load_materialized_workflow
from molexp.agent.modes.run.repair import RuntimeFailure
from molexp.agent.modes.run.run_folder import RunReport
from molexp.agent.modes.run.stages import (
    ExecuteWorkflow,
    LoadMaterializedWorkflow,
    RepairRuntimeFailure,
)
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff
    from molexp.agent.modes.plan.plan_folder import PlanFolder
    from molexp.workflow import Workflow
    from molexp.workspace import Experiment

_LOG = get_logger(__name__)

# ``load_materialized_workflow`` is re-exported so the LoadMaterializedWorkflow
# stage looks it up via this module — preserves existing
# ``monkeypatch.setattr(mode_module, "load_materialized_workflow", ...)``
# test patterns across the substrate migration.
__all__ = ["RunMode", "RunModeConfig", "load_materialized_workflow"]


class RunModeConfig(BaseModel):
    """Tunables for :class:`RunMode`."""

    model_config = ConfigDict(frozen=True)

    retry_backoff_seconds: float = 0.0
    max_repair_escalations: int = 1
    require_execution_gate: bool = True


_CLASS_PIPELINE = ModePipeline(
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


class RunMode(AgentMode):
    """Execute, monitor, and repair an AuthorMode-materialized workflow."""

    name = "run"
    pipeline = _CLASS_PIPELINE

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
        # Per-run scratch — stages mutate, ``_completion`` reads.
        self._workflow: Workflow | None = None
        self._load_error: WorkflowLoadError | None = None
        self._execution_outcome: ExecutionOutcome | None = None
        self._final_report: RunReport | None = None
        self._final_runtime_failure: RuntimeFailure | None = None
        self._final_terminal_state: PlanState | None = None
        # Pipeline tweaked from the class-level placeholder: real Stage
        # subclasses bound to this mode instance. Note: the executor's
        # default routing follows the first edge ("completed" for
        # ExecuteWorkflow), so the success path leaves
        # RepairRuntimeFailure unrun by the executor. We rebuild
        # ``edges`` so all three stages run sequentially per phase 03's
        # always-run RepairRuntimeFailure design: the third stage is
        # an "always-final" decider that branches on outcome.
        self.pipeline = ModePipeline(
            stages=(
                LoadMaterializedWorkflow(run_mode=self),
                ExecuteWorkflow(run_mode=self),
                RepairRuntimeFailure(run_mode=self),
            ),
            entry="LoadMaterializedWorkflow",
            edges=(
                PipelineEdge(from_stage="LoadMaterializedWorkflow", to_stage="ExecuteWorkflow"),
                PipelineEdge(from_stage="ExecuteWorkflow", to_stage="RepairRuntimeFailure"),
                PipelineEdge(from_stage="RepairRuntimeFailure", to_stage="completed"),
            ),
            terminal_states=("completed", "repair_escalated"),
        )

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

        gate_cleared = await self._clear_execution_gate(harness, handoff)
        if not gate_cleared:
            yield self._completion(
                harness,
                text=f"Execution of plan {handoff.plan_id} was not approved.",
                terminal_state=self.plan_folder.plan_state,
            )
            return

        self._transition(PlanState.running)

        # Reset per-run scratch.
        self._workflow = None
        self._load_error = None
        self._execution_outcome = None
        self._final_report = None
        self._final_runtime_failure = None
        self._final_terminal_state = None

        async for event in self.run_pipeline(
            harness=harness,
            user_input=user_input,
            initial_input=handoff,
        ):
            yield event

        # Build the terminal completion event from stage-set state.
        if self._load_error is not None:
            yield self._completion(
                harness,
                text=f"RunMode could not load the materialized workflow: {self._load_error}",
                terminal_state=PlanState.failed,
            )
            return
        if self._execution_outcome is not None and self._execution_outcome.succeeded:
            yield self._completion(
                harness,
                text=(
                    f"Plan {handoff.plan_id} executed — "
                    f"{len(self._execution_outcome.progress.steps)} step(s) completed."
                ),
                terminal_state=PlanState.completed,
                report=self._final_report,
            )
            return
        # Failure path — RepairRuntimeFailure stage set _final_terminal_state.
        terminal_state = self._final_terminal_state or PlanState.failed
        text = self._failure_text(handoff.plan_id, self._final_runtime_failure, terminal_state)
        yield self._completion(
            harness,
            text=text,
            terminal_state=terminal_state,
            report=self._final_report,
        )

    async def _clear_execution_gate(
        self, harness: AgentHarness, handoff: MaterializedWorkspaceHandoff
    ) -> bool:
        if not self.config.require_execution_gate:
            return True
        decision = await approve_execution_gate(handoff, harness=harness)
        return decision.approved

    def _transition(self, dst: PlanState) -> None:
        if self.plan_folder.plan_state is dst:
            return
        self.plan_folder.transition_to(dst)
        self.plan_folder.save()

    def _safe_transition(self, dst: PlanState) -> None:
        """Best-effort transition; swallow an illegal jump (stage uses this)."""
        try:
            self._transition(dst)
        except IllegalPlanTransitionError:
            _LOG.warning(
                f"[run] cannot transition {self.plan_folder.plan_state.value} -> "
                f"{dst.value}; leaving plan state unchanged"
            )

    @staticmethod
    def _failure_text(
        plan_id: str, failure: RuntimeFailure | None, terminal_state: PlanState
    ) -> str:
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
        harness.session.append_message(Message(role="assistant", content=text))
        mode_state: dict[str, object] = {"plan_state": terminal_state.value}
        if report is not None:
            mode_state["run"] = report.model_dump(mode="json")
        result = AgentRunResult(
            text=text,
            messages=harness.session.build_context(),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        return ModeCompletedEvent(text=text, result=result.model_dump(mode="json"))
