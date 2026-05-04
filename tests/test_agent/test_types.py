"""Phase 1a: core types are JSON-roundtrippable and frozen."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from molexp.agent import (
    AgentFailure,
    AgentMode,
    ArtifactRef,
    FailureKind,
    Goal,
    Message,
    Usage,
    WorkflowPreview,
)


def test_goal_defaults_are_immutable() -> None:
    goal = Goal(description="do the thing")
    assert goal.description == "do the thing"
    assert goal.mode is AgentMode.CHAT
    with pytest.raises(FrozenInstanceError):
        goal.description = "changed"  # type: ignore[misc]


def test_message_round_trips_through_json() -> None:
    msg = Message(role="user", content="hello", metadata={"source": "ui"})
    payload = json.dumps(asdict(msg))
    revived = json.loads(payload)
    assert revived == {
        "role": "user",
        "content": "hello",
        "name": None,
        "metadata": {"source": "ui"},
    }


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


def test_workflow_preview_default_lists() -> None:
    preview = WorkflowPreview(workflow_ir={"nodes": []})
    assert preview.python_script == ""
    assert preview.intervention_points == []


def test_artifact_ref_default_kind() -> None:
    art = ArtifactRef(kind="plot", title="run summary", payload={"spec": {}})
    assert art.kind == "plot"
    assert art.path is None
