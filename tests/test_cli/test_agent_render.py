"""``AgentEventRenderer`` — the CLI's consumer of the AgentEvent stream.

Rendering is CLI-owned (the agent library emits plain typed events and never
imports ``rich``); this file is the sole owner of one render path per event
kind. Each test pins a distinct rendering behavior.
"""

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


class TestAgentEventRenderer:
    def test_every_event_kind_has_a_render_path(self) -> None:
        """Each of the 16 AgentEvent kinds renders without crashing + emits output."""
        samples: list[AgentEvent] = [
            LoopStartedEvent(loop_name="agent", user_input="hi"),
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
        for needle in ("agent", "read_file", "plan-1", "preflight", "boom", "which solvent?"):
            assert needle in out

    def test_streamed_answer_renders_as_markdown(self) -> None:
        """At stream close the buffered answer is re-rendered as markdown."""
        out = _render([TokenDeltaEvent(text="# Resu"), TokenDeltaEvent(text="lt\n\nfine")])
        assert "Result" in out
        assert "# Result" not in out  # the heading marker was interpreted, not echoed

    def test_streamed_answer_not_reprinted_by_loop_completed(self) -> None:
        out = _render(
            [
                LoopStartedEvent(loop_name="agent", user_input="q"),
                TokenDeltaEvent(text="the answer"),
                LoopCompletedEvent(text="the answer"),
            ]
        )
        assert out.count("the answer") == 1

    def test_final_text_prints_when_nothing_streamed(self) -> None:
        """The /plan path streams no token deltas — the final text must still print."""
        out = _render(
            [
                LoopStartedEvent(loop_name="agent", user_input="/plan x"),
                LoopCompletedEvent(text="Planning paused — clarification required."),
            ]
        )
        assert "clarification required" in out

    def test_thinking_deltas_do_not_suppress_final_text(self) -> None:
        """Thinking deltas are not answer tokens — final text must still print."""
        out = _render(
            [
                LoopStartedEvent(loop_name="agent", user_input="q"),
                ThinkingDeltaEvent(text="hmm"),
                LoopCompletedEvent(text="the answer"),
            ]
        )
        assert "the answer" in out

    def test_turn_footer_shows_duration_and_usage(self) -> None:
        out = _render(
            [
                LoopStartedEvent(loop_name="agent", user_input="q", timestamp=_T0),
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

    def test_failed_tool_call_uses_distinct_glyph(self) -> None:
        ok_out = _render([ToolCallCompletedEvent(tool_name="read_file", ok=True)])
        bad_out = _render([ToolCallCompletedEvent(tool_name="read_file", ok=False)])
        assert "✓" in ok_out
        assert "✗" in bad_out
        assert "read_file" in bad_out

    def test_newline_separates_thinking_from_answer_streams(self) -> None:
        """Reasoning and answer are distinct streams — a newline separates them."""
        out = _render([ThinkingDeltaEvent(text="reasoning"), TokenDeltaEvent(text="answer")])
        assert "reasoning" in out
        assert "answer" in out
        assert "reasoninganswer" not in out  # not concatenated onto one line

    def test_finish_is_idempotent(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, width=100, force_terminal=False)
        renderer = AgentEventRenderer(console)
        renderer.render(TokenDeltaEvent(text="partial"))
        renderer.finish()
        renderer.finish()  # second call must not reprint or raise
        assert buffer.getvalue().count("partial") == 1
