"""``AgentEventRenderer`` — one render path per AgentEvent kind."""

from __future__ import annotations

import io

from rich.console import Console

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
from molexp.cli.agent_render import AgentEventRenderer


def _render(events: list[AgentEvent]) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    renderer = AgentEventRenderer(console)
    for event in events:
        renderer.render(event)
    return buffer.getvalue()


def test_renders_every_event_kind_without_crashing() -> None:
    """Each of the 15 AgentEvent kinds has a render path and produces output."""
    samples: list[AgentEvent] = [
        ModeStartedEvent(mode_name="interactive", user_input="hi"),
        StageStartedEvent(stage_name="agentic-loop"),
        TokenDeltaEvent(text="hello "),
        ToolCallStartedEvent(tool_name="read_file", args_summary="path=a.py"),
        ToolCallCompletedEvent(tool_name="read_file", result_summary="12 lines", ok=True),
        ArtifactWrittenEvent(path="out.json", description="a plan"),
        ApprovalRequestedEvent(gate="approve_direction", summary="ship it?"),
        ApprovalDecidedEvent(gate="approve_direction", approved=True, reason="ok"),
        PlanEmittedEvent(plan_id="plan-1", step_count=4),
        PreflightFailedEvent(failed_checks=("acyclic",)),
        RepairProposedEvent(failed_invariant="dag", rationale="fix it"),
        CompactionPerformedEvent(summary="...", tokens_before=99, entries_summarized=3),
        StageCompletedEvent(stage_name="agentic-loop"),
        ErrorEvent(message="boom", error_type="ValueError", stage_name="agentic-loop"),
        ModeCompletedEvent(text="all done"),
    ]
    out = _render(samples)
    assert out.strip()
    # spot-check a few kind-specific substrings survived rendering
    for needle in ("interactive", "read_file", "plan-1", "preflight", "boom"):
        assert needle in out


def test_token_deltas_stream_inline() -> None:
    out = _render([TokenDeltaEvent(text="abc"), TokenDeltaEvent(text="def")])
    assert "abcdef" in out


def test_mode_completed_after_token_deltas_does_not_double_print() -> None:
    out = _render(
        [
            ModeStartedEvent(mode_name="interactive", user_input="q"),
            TokenDeltaEvent(text="the answer"),
            ModeCompletedEvent(text="the answer"),
        ]
    )
    assert out.count("the answer") == 1


def test_mode_completed_without_token_deltas_prints_text() -> None:
    """The /plan path streams no token deltas — the final text must print."""
    out = _render(
        [
            ModeStartedEvent(mode_name="interactive", user_input="/plan x"),
            ModeCompletedEvent(text="Planning paused — clarification required."),
        ]
    )
    assert "clarification required" in out


def test_failed_tool_call_marked_distinctly() -> None:
    out = _render([ToolCallCompletedEvent(tool_name="read_file", ok=False)])
    assert "read_file" in out
