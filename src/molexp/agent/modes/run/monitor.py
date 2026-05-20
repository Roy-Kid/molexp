"""Project workflow execution onto the typed plan — ``StepMonitor``.

RunMode executes a materialized :class:`molexp.workflow.Workflow`, but the
UI and the review queue read the *typed plan* — the
:class:`~molexp.agent.modes._planning.PlanGraph` of
:class:`~molexp.agent.modes._planning.PlanStep`\\ s. :class:`StepMonitor`
bridges the two: it projects a :class:`molexp.workflow.WorkflowResult`
onto the plan, producing one :class:`StepProgress` row per ``PlanStep``
and folding them into a frozen :class:`RunProgress`.

The projection is per-step-id: a plan step whose ``id`` appears in the
:attr:`WorkflowResult.outputs` map is ``succeeded``; a step absent from a
*failed* result is ``failed``; a step absent from a *completed* result is
``skipped`` (it never ran — usually a downstream of a failed step).

:class:`StepProgress` / :class:`RunProgress` are frozen-pydantic value
types; :class:`StepMonitor` is a plain runtime class — it accumulates the
live per-step rows as execution proceeds.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.modes._planning import PlanGraph
from molexp.agent.types import utc_now

__all__ = [
    "RunProgress",
    "StepMonitor",
    "StepProgress",
    "StepStatus",
]


class StepStatus(StrEnum):
    """Execution status of one projected plan step."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class StepProgress(BaseModel):
    """Execution progress of one :class:`PlanStep`, projected from a run.

    Attributes:
        step_id: ``id`` of the projected :class:`PlanStep`.
        status: The :class:`StepStatus` of the step.
        started_at: When the step began, or ``None`` if it never started.
        finished_at: When the step finished, or ``None``.
        attempts: How many attempts the step consumed (``>= 0``).
        stdout_ref: Workspace-relative reference to captured stdout, or
            ``None``.
        stderr_ref: Workspace-relative reference to captured stderr, or
            ``None``.
        error_ref: Reference to a captured error trace, or ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str
    status: StepStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int = Field(default=0, ge=0)
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    error_ref: str | None = None


class RunProgress(BaseModel):
    """The typed progress snapshot RunMode emits — one row per plan step.

    This is the shape the UI and the review queue consume; both read the
    same typed plan RunMode executed.

    Attributes:
        plan_id: The plan being executed.
        steps: One :class:`StepProgress` per :class:`PlanStep`, in plan
            order.
        captured_at: When the snapshot was taken.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    steps: tuple[StepProgress, ...]
    captured_at: datetime = Field(default_factory=utc_now)

    @property
    def all_succeeded(self) -> bool:
        """Return whether every projected step succeeded."""
        return bool(self.steps) and all(s.status is StepStatus.succeeded for s in self.steps)

    @property
    def failed_step_ids(self) -> tuple[str, ...]:
        """Return the ids of steps whose status is ``failed``, in plan order."""
        return tuple(s.step_id for s in self.steps if s.status is StepStatus.failed)

    def step(self, step_id: str) -> StepProgress | None:
        """Return the :class:`StepProgress` for ``step_id``, or ``None``."""
        for row in self.steps:
            if row.step_id == step_id:
                return row
        return None


class StepMonitor:
    """Project a workflow execution onto the typed plan.

    A plain runtime class — it accumulates per-step :class:`StepProgress`
    rows as execution proceeds and folds them into a :class:`RunProgress`.

    Construct it with the :class:`PlanGraph` whose steps are being
    executed; call :meth:`mark_running` / :meth:`mark_succeeded` /
    :meth:`mark_failed` / :meth:`mark_skipped` as the executor learns each
    step's fate, then :meth:`snapshot` to fold the accumulated rows into a
    frozen :class:`RunProgress`.
    """

    def __init__(self, plan_graph: PlanGraph) -> None:
        self._plan_id = plan_graph.plan_id
        self._order: tuple[str, ...] = tuple(s.id for s in plan_graph.steps)
        self._rows: dict[str, StepProgress] = {
            step.id: StepProgress(step_id=step.id, status=StepStatus.pending)
            for step in plan_graph.steps
        }

    def mark_running(self, step_id: str, *, attempts: int = 1) -> None:
        """Record that ``step_id`` has started."""
        self._update(
            step_id,
            status=StepStatus.running,
            started_at=utc_now(),
            attempts=attempts,
        )

    def mark_succeeded(self, step_id: str, *, attempts: int = 1) -> None:
        """Record that ``step_id`` finished successfully."""
        self._update(
            step_id,
            status=StepStatus.succeeded,
            finished_at=utc_now(),
            attempts=attempts,
        )

    def mark_failed(
        self,
        step_id: str,
        *,
        attempts: int = 1,
        stdout_ref: str | None = None,
        stderr_ref: str | None = None,
        error_ref: str | None = None,
    ) -> None:
        """Record that ``step_id`` failed, with optional capture references."""
        self._update(
            step_id,
            status=StepStatus.failed,
            finished_at=utc_now(),
            attempts=attempts,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            error_ref=error_ref,
        )

    def mark_skipped(self, step_id: str) -> None:
        """Record that ``step_id`` never ran (a downstream of a failed step)."""
        self._update(step_id, status=StepStatus.skipped)

    def snapshot(self) -> RunProgress:
        """Fold the accumulated rows into a frozen :class:`RunProgress`."""
        return RunProgress(
            plan_id=self._plan_id,
            steps=tuple(self._rows[step_id] for step_id in self._order),
        )

    def _update(self, step_id: str, **changes: object) -> None:
        """Replace ``step_id``'s row with an updated copy (immutable)."""
        if step_id not in self._rows:
            raise KeyError(f"StepMonitor: unknown step id {step_id!r}")
        self._rows[step_id] = self._rows[step_id].model_copy(update=changes)
