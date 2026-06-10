"""``AgentEventRenderer`` — one render path per AgentEvent kind."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

from rich.console import Console

from molexp.agent.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    ArtifactWrittenEvent,
    ClarificationRequiredEvent,
    CompactionPerformedEvent,
    ErrorEvent,
    LoopCompletedEvent,
    LoopStartedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.cli.agent_render import AgentEventRenderer

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _render(events: list[AgentEvent]) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    renderer = AgentEventRenderer(console)
    for event in events:
        renderer.render(event)
    renderer.finish()  # the REPL closes every turn in a ``finally``
    return buffer.getvalue()


def test_renders_every_event_kind_without_crashing() -> None:
    """Each of the 16 AgentEvent kinds has a render path and produces output."""
    samples: list[AgentEvent] = [
        LoopStartedEvent(loop_name="interactive", user_input="hi"),
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
        ClarificationRequiredEvent(questions="which solvent?"),
        CompactionPerformedEvent(summary="...", tokens_before=99, entries_summarized=3),
        StageCompletedEvent(stage_name="agentic-loop"),
        ErrorEvent(message="boom", error_type="ValueError", stage_name="agentic-loop"),
        LoopCompletedEvent(text="all done"),
    ]
    out = _render(samples)
    assert out.strip()
    # spot-check a few kind-specific substrings survived rendering
    for needle in ("interactive", "read_file", "plan-1", "preflight", "boom", "which solvent?"):
        assert needle in out


def test_token_deltas_stream_inline() -> None:
    out = _render([TokenDeltaEvent(text="abc"), TokenDeltaEvent(text="def")])
    assert "abcdef" in out


def test_streamed_answer_renders_as_markdown() -> None:
    """At stream close the buffered answer is re-rendered as markdown."""
    out = _render(
        [
            TokenDeltaEvent(text="# Resu"),
            TokenDeltaEvent(text="lt\n\nfine"),
        ]
    )
    assert "Result" in out
    assert "# Result" not in out  # the heading marker was interpreted, not echoed


def test_loop_completed_after_token_deltas_does_not_double_print() -> None:
    out = _render(
        [
            LoopStartedEvent(loop_name="interactive", user_input="q"),
            TokenDeltaEvent(text="the answer"),
            LoopCompletedEvent(text="the answer"),
        ]
    )
    assert out.count("the answer") == 1


def test_loop_completed_without_token_deltas_prints_text() -> None:
    """The /plan path streams no token deltas — the final text must print."""
    out = _render(
        [
            LoopStartedEvent(loop_name="interactive", user_input="/plan x"),
            LoopCompletedEvent(text="Planning paused — clarification required."),
        ]
    )
    assert "clarification required" in out


def test_turn_footer_shows_duration_and_usage() -> None:
    out = _render(
        [
            LoopStartedEvent(loop_name="interactive", user_input="q", timestamp=_T0),
            LoopCompletedEvent(
                text="hi",
                timestamp=_T0 + timedelta(seconds=3.4),
                result={"usage": {"input_tokens": 1200, "output_tokens": 845}},
            ),
        ]
    )
    assert "done" in out
    assert "3.4s" in out
    assert "1.2k" in out
    assert "845" in out


def test_failed_tool_call_marked_distinctly() -> None:
    ok_out = _render([ToolCallCompletedEvent(tool_name="read_file", ok=True)])
    bad_out = _render([ToolCallCompletedEvent(tool_name="read_file", ok=False)])
    assert "✓" in ok_out
    assert "✗" in bad_out
    assert "read_file" in bad_out


def test_tool_call_duration_rendered() -> None:
    out = _render(
        [
            ToolCallStartedEvent(tool_name="search", timestamp=_T0),
            ToolCallCompletedEvent(tool_name="search", timestamp=_T0 + timedelta(seconds=1.2)),
        ]
    )
    assert "1.2s" in out


def test_thinking_deltas_stream_inline() -> None:
    out = _render([ThinkingDeltaEvent(text="weigh"), ThinkingDeltaEvent(text="ing")])
    assert "weighing" in out


def test_thinking_then_token_inserts_newline_between_streams() -> None:
    """Reasoning and answer are distinct streams — a newline separates them."""
    out = _render(
        [
            ThinkingDeltaEvent(text="reasoning"),
            TokenDeltaEvent(text="answer"),
        ]
    )
    assert "reasoning" in out
    assert "answer" in out
    # the answer is not concatenated onto the reasoning line
    assert "reasoninganswer" not in out


def test_thinking_does_not_suppress_final_text() -> None:
    """Thinking deltas are not answer tokens — final text must still print."""
    out = _render(
        [
            LoopStartedEvent(loop_name="interactive", user_input="q"),
            ThinkingDeltaEvent(text="hmm"),
            LoopCompletedEvent(text="the answer"),
        ]
    )
    assert "the answer" in out


def test_clarification_required_renders_questions() -> None:
    out = _render([ClarificationRequiredEvent(questions="what temperature range?")])
    assert "clarification" in out
    assert "what temperature range?" in out


def test_finish_is_idempotent() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    renderer = AgentEventRenderer(console)
    renderer.render(TokenDeltaEvent(text="partial"))
    renderer.finish()
    renderer.finish()  # second call must not reprint or raise
    assert buffer.getvalue().count("partial") == 1
