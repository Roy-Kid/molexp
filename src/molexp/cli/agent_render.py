"""Rich terminal renderer for the agent :data:`AgentEvent` stream.

The renderer lives in the CLI layer on purpose: the agent library emits
plain typed :data:`~molexp.agent.events.AgentEvent`\\ s and never
imports ``rich``. :class:`AgentEventRenderer` is the ``molexp agent``
REPL's *consumer* of that stream — one render path per event kind.

Visual language of a turn::

    ▶ interactive
      ✦ thinking
      <dim reasoning stream…>
      ⚙ read_file (path=a.py)
      ✓ read_file · 12 lines · 0.8s
    <answer rendered as markdown>
    ╰ done · 3.4s · ↑1.2k ↓845 tok

Two inline streams exist — reasoning (``thinking_delta``) and the
answer (``token_delta``). Reasoning streams dimmed as raw text; the
answer is buffered and, on a terminal, followed live in a transient
tail window, then re-rendered once as full :class:`~rich.markdown.Markdown`
when the stream closes (so code fences, lists and headers come out
formatted). Decision points (approvals, clarifications, errors) render
as bordered panels; everything else is a one-line glyph entry. Durations
are derived from the event ``timestamp``\\ s — no clocks of its own.

Callers must invoke :meth:`AgentEventRenderer.finish` when a turn ends
(the REPL does so in a ``finally``) so an interrupted stream never
leaves a live region open or buffered answer text unprinted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

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

__all__ = ["AgentEventRenderer"]

_STREAM_KINDS = frozenset({"token_delta", "thinking_delta"})
"""Event kinds that stream inline without a trailing newline per delta."""

_INDENT = "  "
"""Left gutter for activity one-liners under the ``▶`` turn header."""

_LIVE_WINDOW_MARGIN = 6
"""Terminal rows reserved around the live answer tail (header, prompt…)."""


def _fmt_duration(seconds: float) -> str:
    """Format a duration compactly: ``0.8s`` / ``42s`` / ``1m07s``."""
    if seconds < 0:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m{rest:02d}s"


def _fmt_tokens(count: int) -> str:
    """Format a token count compactly: ``845`` / ``1.2k`` / ``3.4M``."""
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


class AgentEventRenderer:
    """Render a live :data:`AgentEvent` stream to a rich console.

    Construct one per REPL session; :meth:`render` is called once per
    event in emission order, :meth:`finish` once per turn (idempotent).
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._stream_mode: str | None = None
        self._saw_token_delta = False
        self._answer_buf: list[str] = []
        self._live: Live | None = None
        self._turn_started: datetime | None = None
        self._tool_started: dict[str, list[datetime]] = {}
        self._stage_started: dict[str, datetime] = {}

    def render(self, event: AgentEvent) -> None:
        """Render one event, dispatching on its discriminator ``kind``."""
        # Reasoning ("thinking_delta") and answer ("token_delta") are two
        # distinct inline streams: close the current one on any non-stream
        # event, or when switching from one stream mode to the other, so a
        # newline separates reasoning from the answer it precedes.
        if event.kind not in _STREAM_KINDS or event.kind != self._stream_mode:
            self._end_stream()
        handler = getattr(self, f"_render_{event.kind}", None)
        if handler is None:  # pragma: no cover - dispatch is exhaustive
            self.console.print(Text(f"{_INDENT}· {event.kind}", style="dim"))
            return
        handler(event)

    def finish(self) -> None:
        """Close any in-progress stream; safe to call more than once.

        The REPL calls this in a ``finally`` so an exception mid-stream
        never leaves a live region open or answer text unprinted.
        """
        self._end_stream()

    # ── stream lifecycle ─────────────────────────────────────────────────

    def _end_stream(self) -> None:
        """Close the in-progress inline stream, flushing buffered text."""
        if self._stream_mode == "thinking_delta":
            self.console.print()  # newline closing the dim reasoning line
        self._flush_answer()
        self._stream_mode = None

    def _flush_answer(self) -> None:
        """Stop the live tail window and print the full answer as markdown."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        text = "".join(self._answer_buf)
        self._answer_buf.clear()
        if text.strip():
            self.console.print(Markdown(text))

    def _ensure_live(self) -> Live:
        """Start (once) the transient tail window following the answer."""
        if self._live is None:
            self._live = Live(console=self.console, transient=True, refresh_per_second=12)
            self._live.start()
        return self._live

    def _answer_tail(self) -> Text:
        """Last screenful of the buffered answer, as plain text."""
        window = max(3, self.console.size.height - _LIVE_WINDOW_MARGIN)
        tail = "".join(self._answer_buf).split("\n")[-window:]
        return Text("\n".join(tail))

    # ── per-kind render paths ────────────────────────────────────────────

    def _render_loop_started(self, event: LoopStartedEvent) -> None:
        self._saw_token_delta = False
        self._turn_started = event.timestamp
        self._tool_started.clear()
        self._stage_started.clear()
        self.console.print(Text.assemble(("▶ ", "bold cyan"), (event.loop_name, "bold")))

    def _render_thinking_delta(self, event: ThinkingDeltaEvent) -> None:
        # Reasoning streams dimmed/italic under a one-time label, so it
        # reads as private chain-of-thought rather than the answer.
        if self._stream_mode != "thinking_delta":
            self.console.print(Text(f"{_INDENT}✦ thinking", style="dim italic"))
        self.console.print(
            event.text, end="", style="dim italic", soft_wrap=True, markup=False, highlight=False
        )
        self._stream_mode = "thinking_delta"

    def _render_token_delta(self, event: TokenDeltaEvent) -> None:
        self._answer_buf.append(event.text)
        self._saw_token_delta = True
        if self.console.is_terminal:
            self._ensure_live().update(self._answer_tail())
        self._stream_mode = "token_delta"

    def _render_tool_call_started(self, event: ToolCallStartedEvent) -> None:
        self._tool_started.setdefault(event.tool_name, []).append(event.timestamp)
        line = Text.assemble((f"{_INDENT}⚙ ", "yellow"), (event.tool_name, "bold"))
        if event.args_summary:
            line.append(f" ({event.args_summary})", style="dim")
        self.console.print(line)

    def _render_tool_call_completed(self, event: ToolCallCompletedEvent) -> None:
        duration = self._pop_tool_duration(event.tool_name, event.timestamp)
        if event.ok:
            line = Text.assemble((f"{_INDENT}✓ ", "green"), (event.tool_name, "dim"))
            if event.result_summary:
                line.append(f" · {event.result_summary}", style="dim")
        else:
            line = Text.assemble((f"{_INDENT}✗ ", "bold red"), (event.tool_name, "red"))
            if event.result_summary:
                line.append(f" · {event.result_summary}", style="red")
        if duration:
            line.append(f" · {duration}", style="dim cyan")
        self.console.print(line)

    def _pop_tool_duration(self, tool_name: str, completed_at: datetime) -> str:
        """Duration since the matching start event, or ``""`` when unknown."""
        stack = self._tool_started.get(tool_name)
        if not stack:
            return ""
        return _fmt_duration((completed_at - stack.pop()).total_seconds())

    def _render_stage_started(self, event: StageStartedEvent) -> None:
        self._stage_started[event.stage_name] = event.timestamp
        self.console.print(Text(f"{_INDENT}▸ {event.stage_name}", style="dim"))

    def _render_stage_completed(self, event: StageCompletedEvent) -> None:
        line = Text(f"{_INDENT}▸ {event.stage_name} done", style="dim")
        started = self._stage_started.pop(event.stage_name, None)
        if started is not None:
            line.append(f" · {_fmt_duration((event.timestamp - started).total_seconds())}")
        self.console.print(line)

    def _render_artifact_written(self, event: ArtifactWrittenEvent) -> None:
        line = Text.assemble((f"{_INDENT}⬡ ", "blue"), (event.path, "bold blue"))
        if event.description:
            line.append(f" — {event.description}", style="dim")
        self.console.print(line)

    def _render_approval_requested(self, event: ApprovalRequestedEvent) -> None:
        body = Text(event.summary or "waiting for a decision…")
        title = Text.assemble(("approval · ", "magenta"), (event.gate, "bold magenta"))
        self.console.print(Panel(body, title=title, border_style="magenta", expand=False))

    def _render_approval_decided(self, event: ApprovalDecidedEvent) -> None:
        if event.approved:
            line = Text.assemble((f"{_INDENT}✓ approved", "green"), (f" · {event.gate}", "dim"))
        else:
            line = Text.assemble((f"{_INDENT}✗ rejected", "bold red"), (f" · {event.gate}", "dim"))
        if event.reason:
            line.append(f" ({event.reason})", style="dim")
        self.console.print(line)

    def _render_plan_emitted(self, event: PlanEmittedEvent) -> None:
        steps = f"{event.step_count} step" + ("s" if event.step_count != 1 else "")
        line = Text.assemble(
            (f"{_INDENT}▤ plan ", "cyan"), (event.plan_id, "bold cyan"), (f" · {steps}", "dim")
        )
        self.console.print(line)

    def _render_preflight_failed(self, event: PreflightFailedEvent) -> None:
        checks = ", ".join(event.failed_checks) or "(unnamed)"
        self.console.print(
            Text.assemble((f"{_INDENT}⚠ preflight failed", "bold red"), (f" · {checks}", "red"))
        )

    def _render_repair_proposed(self, event: RepairProposedEvent) -> None:
        line = Text.assemble((f"{_INDENT}⚒ repair ", "yellow"), (event.failed_invariant, "bold"))
        if event.rationale:
            line.append(f" — {event.rationale}", style="dim")
        self.console.print(line)

    def _render_clarification_required(self, event: ClarificationRequiredEvent) -> None:
        self.console.print(
            Panel(
                Text(event.questions),
                title=Text("clarification required", style="bold yellow"),
                border_style="yellow",
                expand=False,
            )
        )

    def _render_compaction_performed(self, event: CompactionPerformedEvent) -> None:
        self.console.print(
            Text(
                f"{_INDENT}≈ compacted {event.entries_summarized} entries"
                f" · {_fmt_tokens(event.tokens_before)} tokens before",
                style="dim",
            )
        )

    def _render_loop_completed(self, event: LoopCompletedEvent) -> None:
        # In the emergent path the final text already streamed as token
        # deltas; only print it when nothing streamed (e.g. the /plan path).
        if not self._saw_token_delta and event.text.strip():
            self.console.print(Markdown(event.text))
        self.console.print(self._turn_footer(event))
        self._turn_started = None

    def _turn_footer(self, event: LoopCompletedEvent) -> Text:
        """Compose the dim ``╰ done · 3.4s · ↑in ↓out tok`` turn footer."""
        parts = ["done"]
        if self._turn_started is not None:
            parts.append(_fmt_duration((event.timestamp - self._turn_started).total_seconds()))
        usage = _usage_of(event.result)
        if usage:
            parts.append(usage)
        return Text("╰ " + " · ".join(parts), style="dim")

    def _render_error(self, event: ErrorEvent) -> None:
        title = Text("error", style="bold red")
        if event.stage_name:
            title.append(f" · {event.stage_name}", style="red")
        body = Text(event.message)
        if event.error_type:
            body.append(f"\n{event.error_type}", style="dim")
        self.console.print(Panel(body, title=title, border_style="red", expand=False))


def _usage_of(result: dict[str, Any] | None) -> str:
    """Token-usage fragment from a ``loop_completed`` result, or ``""``."""
    usage = (result or {}).get("usage")
    if not isinstance(usage, dict):
        return ""
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    if not isinstance(tokens_in, int) or not isinstance(tokens_out, int):
        return ""
    if tokens_in <= 0 and tokens_out <= 0:
        return ""
    return f"↑{_fmt_tokens(tokens_in)} ↓{_fmt_tokens(tokens_out)} tok"
