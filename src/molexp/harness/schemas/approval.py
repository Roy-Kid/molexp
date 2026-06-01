"""Approval-flow schemas (Phase 6).

Three pydantic types:

- :data:`ApprovalIntent` — Literal of the six scenarios that can require
  approval per ``harness-goal.md`` §7.5.
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

__all__ = ["ApprovalDecision", "ApprovalIntent", "ApprovalRequest"]


ApprovalIntent = Literal[
    "agent_inferred_scientific_parameters",
    "full_execution",
    "hpc_submission",
    "large_resource_request",
    "overwrite",
    "final_report",
]


class ApprovalRequest(BaseModel):
    """A single ask for human (or auto-approver) approval."""

    model_config = ConfigDict(frozen=True)

    id: str
    intent: ApprovalIntent
    reason: str
    triggered_by_policy: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ApprovalDecision(BaseModel):
    """The answer to one :class:`ApprovalRequest`."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    granted: bool
    decided_by: str
    decided_at: datetime
    reason: str | None = None
