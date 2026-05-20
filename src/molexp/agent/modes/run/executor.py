"""Bind a materialized :class:`Workflow` to a workspace ``Run`` and drive it.

:class:`RunExecutor` is the runtime container that actually executes the
LLM-authored workflow. It binds the loaded :class:`molexp.workflow.Workflow`
to a workspace :class:`~molexp.workspace.Run`, enters the run's
:class:`~molexp.workspace.RunContext`, and calls
:meth:`Workflow.execute(run_context=...) <molexp.workflow.Workflow.execute>`
— purely the public ``molexp.workflow`` API; no ``pydantic_graph``.

The execution-record / asset-lineage side effects land on the workspace
``Run`` and ``AssetCatalog`` for free — that is the existing workspace
execution model. :class:`RunExecutor` adds only the projection onto the
typed plan: it drives a :class:`~molexp.agent.modes.run.monitor.StepMonitor`
and returns a frozen :class:`ExecutionOutcome`.

A plain runtime class — it carries a live workspace ``Run`` and an
``asyncio``-driven workflow — per the agent-layer charter (a runtime
container, not pure data).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.modes.run.monitor import RunProgress, StepMonitor, StepStatus

if TYPE_CHECKING:
    from molexp.agent.modes._planning import PlanGraph
    from molexp.workflow import Workflow, WorkflowResult
    from molexp.workflow.protocols import RunContextLike
    from molexp.workspace import Run

_LOG = get_logger(__name__)

__all__ = ["ExecutionOutcome", "RunExecutor"]


class ExecutionOutcome(BaseModel):
    """The frozen result of one :meth:`RunExecutor.execute` call.

    Attributes:
        status: The workflow's terminal status — ``"completed"`` /
            ``"failed"`` / ``"cancelled"``.
        progress: The :class:`RunProgress` projected onto the typed plan.
        run_id: The workspace ``Run`` id the workflow executed against.
        execution_id: The workflow execution id, or ``None``.
        error_type: The exception class name when the run failed, else
            ``""``.
        error_message: The failure message when the run failed, else
            ``""``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    progress: RunProgress
    run_id: str | None = None
    execution_id: str | None = None
    error_type: str = ""
    error_message: str = ""

    @property
    def succeeded(self) -> bool:
        """Return whether the workflow reached a ``completed`` status."""
        return self.status == "completed"


class RunExecutor:
    """Bind a :class:`Workflow` to a workspace ``Run`` and execute it.

    Construct with the loaded :class:`Workflow`, the workspace ``Run`` to
    bind it to, and the :class:`PlanGraph` whose steps the run is
    projected onto. :meth:`execute` enters the run context, drives the
    workflow through the public API, projects the result onto a
    :class:`StepMonitor`, and returns a frozen :class:`ExecutionOutcome`.
    """

    def __init__(
        self,
        *,
        workflow: Workflow,
        run: Run,
        plan_graph: PlanGraph,
    ) -> None:
        self._workflow = workflow
        self._run = run
        self._plan_graph = plan_graph
        self._monitor = StepMonitor(plan_graph)

    @property
    def monitor(self) -> StepMonitor:
        """The :class:`StepMonitor` accumulating per-step projection rows."""
        return self._monitor

    async def execute(self) -> ExecutionOutcome:
        """Run the bound workflow and project the result onto the typed plan.

        Enters the workspace ``Run``'s :class:`RunContext`, calls
        :meth:`Workflow.execute`, and folds the
        :class:`~molexp.workflow.WorkflowResult` into a
        :class:`RunProgress` via the :class:`StepMonitor`.

        A workflow that raises before producing a result is recorded as a
        ``failed`` :class:`ExecutionOutcome` — RunMode classifies and
        repairs from there; the exception does not escape.
        """
        for step in self._plan_graph.steps:
            self._monitor.mark_running(step.id)

        try:
            result = await self._drive_workflow()
        except Exception as exc:
            _LOG.error(f"[run-executor] workflow raised: {type(exc).__name__}: {exc}")
            return self._failed_outcome(error=exc)

        return self._project_result(result)

    async def _drive_workflow(self) -> WorkflowResult:
        """Enter the run context and execute the workflow through the public API.

        The workspace ``RunContext`` is the duck-typed match for the
        workflow layer's :class:`~molexp.workflow.protocols.RunContextLike`
        protocol — the ``cast`` makes that structural identity explicit
        (the same shape ``Workflow.run_on`` relies on internally).
        """
        with self._run.start() as run_ctx:
            return await self._workflow.execute(run_context=cast("RunContextLike", run_ctx))

    def _project_result(self, result: WorkflowResult) -> ExecutionOutcome:
        """Project a :class:`WorkflowResult` onto the typed plan steps.

        A plan step whose id is in ``result.outputs`` succeeded. On a
        ``completed`` run, any step missing from ``outputs`` is
        ``skipped``; on a non-``completed`` run, missing steps are
        ``failed`` (the run stopped before reaching them).
        """
        completed = result.status == "completed"
        produced = set(result.outputs.keys())
        for step in self._plan_graph.steps:
            if step.id in produced:
                self._monitor.mark_succeeded(step.id)
            elif completed:
                self._monitor.mark_skipped(step.id)
            else:
                self._monitor.mark_failed(step.id)
        return ExecutionOutcome(
            status=result.status,
            progress=self._monitor.snapshot(),
            run_id=result.run_id,
            execution_id=result.execution_id,
        )

    def _failed_outcome(self, *, error: BaseException) -> ExecutionOutcome:
        """Build a ``failed`` :class:`ExecutionOutcome` for a raised workflow."""
        for step in self._plan_graph.steps:
            progress = self._monitor.snapshot().step(step.id)
            if progress is not None and progress.status is StepStatus.running:
                self._monitor.mark_failed(step.id, error_ref=f"{type(error).__name__}: {error}")
        return ExecutionOutcome(
            status="failed",
            progress=self._monitor.snapshot(),
            run_id=self._run.id,
            error_type=type(error).__name__,
            error_message=str(error),
        )
