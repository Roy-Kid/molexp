"""``HarnessEvent`` + ``EventType`` — append-only audit timeline.

Per ``.claude/notes/harness-goal.md`` §4.2: every notable thing the harness
does (stage transitions, artifact writes, agent calls, approvals, policy
checks, …) gets one ``HarnessEvent`` appended to the event log, with a
per-``run_id`` monotonic ``seq``.

Phase 1 ships the **flat** single-class shape; ``type`` is a
``Literal[...]`` discriminator. A future Phase may evolve this into a
discriminated union of variant classes (mirroring
:mod:`molexp.agent.harness.events`) — the flat shape's ``type`` discriminator
is forward-compatible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["EventType", "HarnessEvent"]


EventType = Literal[
    "run_created",
    "run_completed",
    "run_failed",
    "stage_started",
    "stage_completed",
    "stage_failed",
    "artifact_created",
    "artifact_validated",
    "validation_passed",
    "validation_failed",
    "agent_called",
    "agent_completed",
    "agent_failed",
    "tool_called",
    "tool_completed",
    "tool_failed",
    "task_started",
    "task_completed",
    "task_failed",
    "test_started",
    "test_completed",
    "test_failed",
    "approval_requested",
    "approval_granted",
    "approval_rejected",
    "policy_checked",
    "policy_passed",
    "policy_failed",
    "artifact_edge_created",
]


class HarnessEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    run_id: str
    seq: int
    type: EventType
    actor: str
    created_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)

    @field_validator("seq")
    @classmethod
    def _seq_must_be_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("HarnessEvent.seq must be non-negative")
        return value
