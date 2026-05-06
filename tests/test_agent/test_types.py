"""Phase 1a: core types are JSON-roundtrippable and frozen.

Pydantic-based contract: Goal / Message / Usage / AgentFailure are
``BaseModel(frozen=True)``; mutation raises ``pydantic.ValidationError``;
serialization goes through ``model_dump()`` / ``model_dump_json()``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from molexp.agent import (
    AgentFailure,
    AgentMode,
    FailureKind,
    Goal,
    Message,
    Usage,
)


def test_goal_defaults_are_immutable() -> None:
    goal = Goal(description="do the thing")
    assert goal.description == "do the thing"
    assert goal.mode is AgentMode.CHAT
    with pytest.raises(ValidationError):
        goal.description = "changed"  # type: ignore[misc]


def test_message_round_trips_through_json() -> None:
    msg = Message(role="user", content="hello", metadata={"source": "ui"})
    payload = json.dumps(msg.model_dump())
    revived = json.loads(payload)
    assert revived == {
        "role": "user",
        "content": "hello",
        "name": None,
        "metadata": {"source": "ui"},
    }


def test_message_validate_round_trip() -> None:
    msg = Message(role="user", content="hi", metadata={"k": "v"})
    revived = Message.model_validate_json(msg.model_dump_json())
    assert revived == msg


def test_usage_is_zero_by_default() -> None:
    u = Usage()
    assert u.total_tokens == 0
    assert u.requests == 0


def test_failure_carries_kind_and_detail() -> None:
    failure = AgentFailure(
        kind=FailureKind.TOOL_NOT_FOUND,
        message="tool 'noop' not registered",
        detail={"requested": "noop"},
    )
    assert failure.kind is FailureKind.TOOL_NOT_FOUND
    assert failure.detail == {"requested": "noop"}


def test_failure_is_frozen() -> None:
    failure = AgentFailure(kind=FailureKind.MODEL_ERROR, message="boom")
    with pytest.raises(ValidationError):
        failure.message = "tampered"  # type: ignore[misc]


def test_artifact_ref_is_gone() -> None:
    """ArtifactRef must not be importable from molexp.agent."""
    import molexp.agent as agent

    assert not hasattr(agent, "ArtifactRef")
    assert "ArtifactRef" not in agent.__all__


def test_workflow_preview_is_gone() -> None:
    """WorkflowPreview must not be importable from molexp.agent."""
    import molexp.agent as agent

    assert not hasattr(agent, "WorkflowPreview")
    assert "WorkflowPreview" not in agent.__all__
