"""RunMode's three Stage subclasses.

- :class:`LoadMaterializedWorkflow` — load the workflow module
- :class:`ExecuteWorkflow` — bind to a workspace ``Run`` and execute
- :class:`RepairRuntimeFailure` — classify failure / persist report /
  emit terminal completion (always runs; on success it's a no-op).

Stages store intermediate results on the bound :class:`RunMode`
instance; :meth:`RunMode._completion` reads them post-pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.events import (
    AgentEvent,
    ErrorEvent,
    RepairProposedEvent,
)
from molexp.agent.modes._planning import IllegalPlanTransitionError, PlanState
from molexp.agent.modes.run.executor import RunExecutor
from molexp.agent.modes.run.loader import WorkflowLoadError
from molexp.agent.modes.run.repair import (
    RuntimeFailure,
    RuntimeFailureKind,
    build_repair_diff,
    build_repair_escalation,
)
from molexp.agent.modes.run.run_folder import RunFolder, RunReport
from molexp.agent.stage import Stage

if TYPE_CHECKING:
    from molexp.agent.modes._planning import PlanDiff, PlanGraph
    from molexp.agent.modes.run._mode import RunMode
    from molexp.agent.modes.run.executor import ExecutionOutcome
    from molexp.agent.modes.run.repair import RepairEscalation
    from molexp.agent.runtime import AgentHarness

__all__ = ["ExecuteWorkflow", "LoadMaterializedWorkflow", "RepairRuntimeFailure"]


class LoadMaterializedWorkflow(Stage[object, object]):
    """Stage 1 — load the materialized workflow module.

    On :class:`WorkflowLoadError`: emits :class:`ErrorEvent`, sets
    ``run_mode._load_error``, and transitions ``plan_folder`` to
    ``failed``. Subsequent stages check ``_load_error`` and no-op.
    """

    name: ClassVar[str] = "LoadMaterializedWorkflow"

    def __init__(self, *, run_mode: RunMode) -> None:
        self._mode = run_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        handoff = mode._injected_handoff
        if handoff is None:
            yield None
            return
        # Look up ``load_materialized_workflow`` via the _mode module so
        # existing ``monkeypatch.setattr(mode_module, ...)`` test patterns
        # are observable inside the stage.
        from molexp.agent.modes.run import _mode as _mode_module

        try:
            workflow = _mode_module.load_materialized_workflow(handoff)
        except WorkflowLoadError as exc:
            mode._load_error = exc
            mode._safe_transition(PlanState.failed)
            await harness.emit(
                ErrorEvent(
                    message=str(exc),
                    error_type=type(exc).__name__,
                    stage_name="load",
                )
            )
            yield None
            return
        mode._workflow = workflow
        yield None


class ExecuteWorkflow(Stage[object, object]):
    """Stage 2 — bind workflow to a workspace ``Run`` and execute."""

    name: ClassVar[str] = "ExecuteWorkflow"

    def __init__(self, *, run_mode: RunMode) -> None:
        self._mode = run_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — substrate contract
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if mode._load_error is not None or mode._workflow is None:
            yield None
            return
        handoff = mode._injected_handoff
        assert handoff is not None  # narrowed by LoadMaterializedWorkflow
        run = mode.experiment.add_run(id=f"run-{handoff.plan_id}")
        executor = RunExecutor(
            workflow=mode._workflow,
            run=run,
            plan_graph=handoff.plan_graph,
        )
        mode._execution_outcome = await executor.execute()
        yield None


class RepairRuntimeFailure(Stage[object, object]):
    """Stage 3 — finalize: on success persist the success report; on
    failure classify, build :class:`PlanDiff` / :class:`RepairEscalation`,
    yield :class:`RepairProposedEvent`, and persist a failure report.

    Always runs (per the executor's sequential routing) — on a clean
    success it just persists the success report and yields no events.
    """

    name: ClassVar[str] = "RepairRuntimeFailure"

    def __init__(self, *, run_mode: RunMode) -> None:
        self._mode = run_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: object,  # noqa: ARG002
    ) -> AsyncIterator[AgentEvent | object]:
        mode = self._mode
        if mode._load_error is not None:
            yield None
            return
        handoff = mode._injected_handoff
        outcome = mode._execution_outcome
        if handoff is None or outcome is None:
            yield None
            return

        if outcome.succeeded:
            mode._transition(PlanState.completed)
            mode._final_report = RunReport(
                plan_id=handoff.plan_id,
                status="completed",
                run_id=outcome.run_id,
                execution_id=outcome.execution_id,
                progress=outcome.progress,
            )
            _persist_report(mode, mode._final_report)
            yield None
            return

        diff, escalation, failure = _build_repair(handoff.plan_graph, outcome)
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
        mode._safe_transition(terminal_state)
        mode._final_runtime_failure = failure
        mode._final_terminal_state = terminal_state
        mode._final_report = RunReport(
            plan_id=handoff.plan_id,
            status=terminal_state.value,
            run_id=outcome.run_id,
            execution_id=outcome.execution_id,
            progress=outcome.progress,
            repair_diffs=(diff,) if diff is not None else (),
            escalation=escalation,
        )
        _persist_report(mode, mode._final_report)
        yield None


def _build_repair(
    plan_graph: PlanGraph,
    outcome: ExecutionOutcome,
) -> tuple[PlanDiff | None, RepairEscalation | None, RuntimeFailure | None]:
    """Build a :class:`PlanDiff` + :class:`RepairEscalation` for a failure."""
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


def _persist_report(mode: RunMode, report: RunReport) -> None:
    """Persist the :class:`RunReport` through the plan-anchored :class:`RunFolder`."""
    folder = _run_folder(mode, report.plan_id)
    folder.write_run_report(report)
    folder.write_progress("final", report.progress)
    if report.escalation is not None:
        folder.write_repair_escalation(report.escalation)


def _run_folder(mode: RunMode, plan_id: str) -> RunFolder:
    name = f"run-{plan_id}"
    if not mode.plan_folder.has_folder(name, cls=RunFolder):
        mode.plan_folder.add_folder(RunFolder(name=name, plan_id=plan_id))
    return mode.plan_folder.get_folder(name, cls=RunFolder)


# Re-export for the mode's _safe_transition fallback handling.
__all__ = (*__all__, "IllegalPlanTransitionError")
