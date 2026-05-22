"""Rich terminal renderer for the agent :data:`AgentEvent` stream.

The renderer lives in the CLI layer on purpose: the agent library emits
plain typed :data:`~molexp.agent.harness.events.AgentEvent`\\ s and never
imports ``rich``. :class:`AgentEventRenderer` is the ``molexp agent``
REPL's *consumer* of that stream — one render path per event kind.

Token deltas stream inline (no newline); every other kind renders as a
distinct one-liner. The renderer tracks just enough state to close an
in-progress streamed line and to avoid reprinting the final text when
it has already streamed token-by-token.
"""

from __future__ import annotations

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

__all__ = ["AgentEventRenderer"]


class AgentEventRenderer:
    """Render a live :data:`AgentEvent` stream to a rich console.

    Construct one per REPL session; :meth:`render` is called once per
    event in emission order.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._streaming = False
        self._saw_token_delta = False

    def render(self, event: AgentEvent) -> None:
        """Render one event, dispatching on its discriminator ``kind``."""
        if not isinstance(event, TokenDeltaEvent):
            self._end_stream()
        handler = getattr(self, f"_render_{event.kind}", None)
        if handler is None:  # pragma: no cover - dispatch is exhaustive
            self.console.print(f"[dim]· {event.kind}[/dim]")
            return
        handler(event)

    def _end_stream(self) -> None:
        """Close an in-progress streamed token line with a newline."""
        if self._streaming:
            self.console.print()
            self._streaming = False

    # ── per-kind render paths ────────────────────────────────────────────

    def _render_mode_started(self, event: ModeStartedEvent) -> None:
        self._saw_token_delta = False
        self.console.print(f"[bold cyan]▶ {event.mode_name}[/bold cyan]")

    def _render_token_delta(self, event: TokenDeltaEvent) -> None:
        self.console.print(event.text, end="", soft_wrap=True, markup=False, highlight=False)
        self._streaming = True
        self._saw_token_delta = True

    def _render_tool_call_started(self, event: ToolCallStartedEvent) -> None:
        detail = f"({event.args_summary})" if event.args_summary else "()"
        self.console.print(f"[yellow]⚙ {event.tool_name}{detail}[/yellow]")

    def _render_tool_call_completed(self, event: ToolCallCompletedEvent) -> None:
        mark = "[green]✓[/green]" if event.ok else "[red]✗[/red]"
        detail = f" — {event.result_summary}" if event.result_summary else ""
        self.console.print(f"{mark} [dim]{event.tool_name}{detail}[/dim]")

    def _render_stage_started(self, event: StageStartedEvent) -> None:
        self.console.print(f"[dim]· stage {event.stage_name}[/dim]")

    def _render_stage_completed(self, event: StageCompletedEvent) -> None:
        self.console.print(f"[dim]· stage {event.stage_name} done[/dim]")

    def _render_artifact_written(self, event: ArtifactWrittenEvent) -> None:
        note = f" — {event.description}" if event.description else ""
        self.console.print(f"[blue]📄 wrote {event.path}{note}[/blue]")

    def _render_approval_requested(self, event: ApprovalRequestedEvent) -> None:
        self.console.print(f"[magenta]? approval: {event.gate}[/magenta]")
        if event.summary:
            self.console.print(f"  [dim]{event.summary}[/dim]")

    def _render_approval_decided(self, event: ApprovalDecidedEvent) -> None:
        verdict = "[green]approved[/green]" if event.approved else "[red]rejected[/red]"
        reason = f" [dim]({event.reason})[/dim]" if event.reason else ""
        self.console.print(f"  → {verdict} for {event.gate}{reason}")

    def _render_plan_emitted(self, event: PlanEmittedEvent) -> None:
        self.console.print(f"[cyan]plan {event.plan_id} — {event.step_count} step(s)[/cyan]")

    def _render_preflight_failed(self, event: PreflightFailedEvent) -> None:
        checks = ", ".join(event.failed_checks) or "(unnamed)"
        self.console.print(f"[red]preflight failed: {checks}[/red]")

    def _render_repair_proposed(self, event: RepairProposedEvent) -> None:
        rationale = f" [dim]{event.rationale}[/dim]" if event.rationale else ""
        self.console.print(f"[yellow]repair: {event.failed_invariant}[/yellow]{rationale}")

    def _render_compaction_performed(self, event: CompactionPerformedEvent) -> None:
        self.console.print(f"[dim]· compacted {event.entries_summarized} entries[/dim]")

    def _render_mode_completed(self, event: ModeCompletedEvent) -> None:
        # In the emergent path the final text already streamed as token
        # deltas; only print it when nothing streamed (e.g. the /plan path).
        if not self._saw_token_delta and event.text:
            self.console.print(event.text)

    def _render_error(self, event: ErrorEvent) -> None:
        where = f" [{event.stage_name}]" if event.stage_name else ""
        self.console.print(f"[bold red]error{where}:[/bold red] {event.message}")
