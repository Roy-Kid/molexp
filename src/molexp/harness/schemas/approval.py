"""Approval-flow schemas (Phase 6).

Three pydantic types:

- :data:`ApprovalIntent` — Literal of the seven scenarios that can require
  approval per ``harness-goal.md`` §7.5 (``experiment_spec`` gates the
  human review of the ExperimentReport before the plan compiles).
- :class:`ApprovalRequest` — what the harness asks a human (or auto-approver)
  to decide on.
- :class:`ApprovalDecision` — the answer.

The Phase-6 evaluator
(:func:`molexp.harness.policy.evaluate_approval_policy`) emits
:class:`ApprovalRequest` instances; the event-log helpers
(:func:`molexp.harness.policy.record_approval_request`,
:func:`molexp.harness.policy.record_approval_decision`) thread them into
the existing :class:`HarnessEvent` stream.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ApprovalDecision", "ApprovalIntent", "ApprovalRequest", "ApprovalScope"]


ApprovalIntent = Literal[
    "agent_inferred_scientific_parameters",
    "full_execution",
    "hpc_submission",
    "large_resource_request",
    "overwrite",
    "final_report",
    "experiment_spec",
    "approve_experiment_plan",
    "task_intervention",
]


ApprovalScope = Literal["approval_gate", "intervention_request"]
"""Resume-routing discriminator for a suspended :class:`ApprovalRequest`.

* ``approval_gate`` (phase-1, the default) — a pipeline approval gate. Resume
  re-drives the *main* planning session through the stage ledger; the stored
  grant replays past the re-entered gate.
* ``intervention_request`` (phase-2) — a named codegen subagent is stuck and
  asks for human guidance. Resume re-seeds and re-runs that one subagent
  (see :attr:`ApprovalRequest.target_agent_id`), not the whole pipeline.
"""


class ApprovalRequest(BaseModel):
    """A single ask for human (or auto-approver) approval."""

    model_config = ConfigDict(frozen=True)

    id: str
    intent: ApprovalIntent
    reason: str
    triggered_by_policy: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    scope: ApprovalScope = "approval_gate"
    """Which resume path answering this request takes.

    Defaults to ``approval_gate`` so every pre-plan-emergent-07 gate request
    (and every legacy stored row) reads back as a phase-1 approval gate.
    """
    target_agent_id: str | None = None
    """The phase-2 codegen subagent to re-seed on an ``intervention_request``.

    ``None`` for an ``approval_gate`` request; required (non-``None``) when
    ``scope == "intervention_request"`` — resume of an intervention with no
    named subagent fails loud rather than silently re-driving the main session.
    """


class ApprovalDecision(BaseModel):
    """The answer to one :class:`ApprovalRequest`."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    granted: bool
    decided_by: str
    decided_at: datetime
    reason: str | None = None
