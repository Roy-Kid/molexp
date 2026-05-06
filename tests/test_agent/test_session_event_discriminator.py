"""SessionEvent discriminated-union round-trip tests."""

from __future__ import annotations

from datetime import datetime, timezone

from molexp.agent.orchestration.events import (
    SESSION_EVENT_ADAPTER,
    ContextBuilt,
    FailureRecorded,
    ModelRequested,
    ModelResponded,
    PlanCreated,
    PlanDecided,
    SessionCompleted,
    SessionEvent,
    SessionStarted,
    ToolApprovalRequested,
    ToolCallCompleted,
    ToolCallRequested,
    TurnStarted,
    UserMessageReceived,
    UserMessageRequested,
)
from molexp.agent.types import AgentFailure, FailureKind, Usage
from molexp.workflow import WorkflowPreviewView

_TS = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)

EVENT_FIXTURES: tuple[SessionEvent, ...] = (
    SessionStarted(session_id="s", goal_description="g", ts=_TS),
    TurnStarted(session_id="s", turn_id="t", ts=_TS),
    ContextBuilt(turn_id="t", used_chars=42, ts=_TS),
    ModelRequested(turn_id="t", model_name="fake", ts=_TS),
    ModelResponded(turn_id="t", finish_reason="stop", usage=Usage(), ts=_TS),
    ToolCallRequested(turn_id="t", call_id="c", tool_name="x", arguments={}, ts=_TS),
    ToolApprovalRequested(turn_id="t", request_id="r", tool_name="x", arguments={}, ts=_TS),
    ToolCallCompleted(turn_id="t", call_id="c", tool_name="x", ok=True, ts=_TS),
    PlanCreated(
        turn_id="t",
        request_id="r",
        plan_markdown="# plan",
        workflow_preview=WorkflowPreviewView(),
        ts=_TS,
    ),
    PlanDecided(request_id="r", approved=True, ts=_TS),
    UserMessageRequested(request_id="r", prompt="?", ts=_TS),
    UserMessageReceived(content="ok", ts=_TS),
    FailureRecorded(
        turn_id="t",
        failure=AgentFailure(kind=FailureKind.MODEL_ERROR, message="boom"),
        ts=_TS,
    ),
    SessionCompleted(session_id="s", summary="done", ts=_TS),
)


class TestSessionEventDiscriminator:
    def test_all_14_event_types_present(self):
        kinds = {e.kind for e in EVENT_FIXTURES}
        assert len(kinds) == 14

    def test_round_trip_preserves_concrete_type(self):
        for event in EVENT_FIXTURES:
            payload = event.model_dump(mode="json")
            revived = SESSION_EVENT_ADAPTER.validate_python(payload)
            assert type(revived) is type(event), (
                f"round-trip lost concrete type: {type(event).__name__} → {type(revived).__name__}"
            )

    def test_unknown_kind_raises(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SESSION_EVENT_ADAPTER.validate_python(
                {"kind": "this_kind_does_not_exist", "session_id": "s"}
            )

    def test_missing_kind_raises(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SESSION_EVENT_ADAPTER.validate_python({"session_id": "s"})
