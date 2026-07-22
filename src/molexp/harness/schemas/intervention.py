"""``BlockedTask`` + ``InterventionRequest`` — the human-in-the-loop payload.

Plan-emergent-06's realization phase maps one self-repairing worker over every
frozen ``BoundTask``. A task that cannot be driven to green within its attempt
budget does not abort the whole board — the worker *returns* a
:class:`BlockedTask` describing the failure, and the board collects every such
block into one durable :class:`InterventionRequest` artifact before signalling
:class:`~molexp.harness.errors.TaskRealizationBlockedError`.

Both are frozen pydantic models (pure data): the request is persisted as an
``intervention_request`` artifact so the approvals inbox can surface it, and a
later human decision (07's concern) can unblock the run.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BlockedTask", "InterventionRequest"]


class BlockedTask(BaseModel):
    """One task the realization phase could not self-repair to green.

    Attributes:
        task_id: The blocked ``BoundTask.id``.
        ir_task_id: The task's originating ``PlanTaskIR.id``.
        slug: The python-module slug derived from ``ir_task_id`` (the generated
            ``workflow/<slug>.py`` / ``tests/test_<slug>.py`` stem).
        attempts: How many codegen+pytest attempts were spent before blocking.
        error: The last failure text (the persisted pytest feedback body — not
            ``str(exc)``), so a human sees the actual traceback.
        question: A natural-language prompt asking how the task should be
            written, shown alongside the failure in the intervention surface.
        error_artifact_id: The persisted failure-feedback artifact id, or
            ``None`` when the failure could not be captured to the store.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    ir_task_id: str
    slug: str
    attempts: int
    error: str
    question: str
    error_artifact_id: str | None = None


class InterventionRequest(BaseModel):
    """A durable "these tasks need a human" record for one bound workflow.

    Attributes:
        id: Unique id for this request (``ivr-…``).
        bound_workflow_id: The ``BoundWorkflow.id`` whose realization stalled.
        blocked_tasks: Every task that exhausted its self-repair budget.
        reason: One-line human summary (e.g. ``"1 task could not be realized"``).
        created_at: When the request was raised (UTC).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    bound_workflow_id: str
    blocked_tasks: tuple[BlockedTask, ...]
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
