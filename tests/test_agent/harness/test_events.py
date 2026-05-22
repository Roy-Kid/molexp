"""``AgentEvent`` discriminated-union tests (spec ac-001)."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from molexp.agent.harness.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    ArtifactWrittenEvent,
    CompactionPerformedEvent,
    ErrorEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

ALL_EVENT_CLASSES = (
    ModeStartedEvent,
    StageStartedEvent,
    StageCompletedEvent,
    ArtifactWrittenEvent,
    ApprovalRequestedEvent,
    ApprovalDecidedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    CompactionPerformedEvent,
    ModeCompletedEvent,
    ErrorEvent,
    TokenDeltaEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
)

EXPECTED_KINDS = {
    "mode_started",
    "stage_started",
    "stage_completed",
    "artifact_written",
    "approval_requested",
    "approval_decided",
    "plan_emitted",
    "preflight_failed",
    "repair_proposed",
    "compaction_performed",
    "mode_completed",
    "error",
    "token_delta",
    "tool_call_started",
    "tool_call_completed",
}


def test_union_covers_all_fifteen_kinds() -> None:
    kinds = {cls.model_fields["kind"].default for cls in ALL_EVENT_CLASSES}
    assert kinds == EXPECTED_KINDS
    assert len(ALL_EVENT_CLASSES) == 15


def test_each_event_carries_a_timestamp() -> None:
    ev = ModeStartedEvent(mode_name="chat", user_input="hi")
    assert isinstance(ev.timestamp, datetime)
    assert ev.timestamp.tzinfo is not None


def test_events_are_frozen() -> None:
    ev = StageStartedEvent(stage_name="draft")
    with pytest.raises(ValidationError):
        ev.stage_name = "other"  # type: ignore[misc]


def test_discriminated_union_round_trips_through_json() -> None:
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    samples: list[AgentEvent] = [
        ModeStartedEvent(mode_name="chat", user_input="hi"),
        StageStartedEvent(stage_name="draft"),
        StageCompletedEvent(stage_name="draft"),
        ArtifactWrittenEvent(path="out.txt", description="result"),
        ApprovalRequestedEvent(gate="approve_direction", summary="check"),
        ApprovalDecidedEvent(gate="approve_direction", approved=True),
        PlanEmittedEvent(plan_id="p1", step_count=3),
        PreflightFailedEvent(failed_checks=("acyclic", "io")),
        RepairProposedEvent(failed_invariant="dag", rationale="fix"),
        CompactionPerformedEvent(summary="...", tokens_before=100, entries_summarized=4),
        ModeCompletedEvent(text="done"),
        ErrorEvent(message="boom", error_type="ValueError"),
        TokenDeltaEvent(text="hel"),
        ToolCallStartedEvent(tool_name="read_file", args_summary="path=a.py"),
        ToolCallCompletedEvent(tool_name="read_file", result_summary="42 lines", ok=True),
    ]
    for ev in samples:
        dumped = adapter.dump_json(ev)
        loaded = adapter.validate_json(dumped)
        assert loaded.kind == ev.kind
        assert type(loaded) is type(ev)


def test_discriminator_selects_concrete_class() -> None:
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    payload = {"kind": "error", "message": "x", "error_type": "RuntimeError"}
    loaded = adapter.validate_python(payload)
    assert isinstance(loaded, ErrorEvent)
    assert loaded.message == "x"


def test_mode_completed_carries_optional_result_payload() -> None:
    ev = ModeCompletedEvent(text="done", result={"mode_state": {"k": 1}})
    assert ev.result == {"mode_state": {"k": 1}}
