"""Cluster 5 — the ``AgentHarness`` runtime object.

:class:`AgentHarness` is the orchestration runtime a mode drives. It
owns the live :class:`~molexp.agent.harness.session.Session`, the
:data:`~molexp.agent.harness.events.EventSink`, the optional
:class:`~molexp.agent.router.Router` (for compaction summarization
only), the optional :class:`~molexp.agent.harness.execution_env.ExecutionEnv`,
and the :class:`~molexp.agent.harness.hooks.HookRegistry`.

It exposes the five capabilities a mode needs:

- :meth:`emit` — push one :data:`AgentEvent` to the sink.
- :meth:`stage` — an async context manager bracketing a unit of work
  with ``stage_started`` / ``stage_completed`` (or ``error``) events
  and the ``before_stage`` / ``after_stage`` hooks.
- :meth:`approve` — the unification point for the three
  :class:`~molexp.agent.modes._planning.ApprovalGate`\\ s: evaluates
  the ``before_approval`` hook and emits ``approval_requested`` then
  ``approval_decided``.
- :meth:`compact` — runs context compaction, routing the summarization
  through :class:`Router` (never ``pydantic_ai``).
- :meth:`run_subprocess` — delegates to the :class:`ExecutionEnv`.

Plain runtime class (it holds live handles), per the agent-layer
charter. Imports nothing from ``pydantic_ai`` / ``pydantic_graph``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from molexp.agent.harness.compaction import CompactionSettings, prepare_compaction
from molexp.agent.harness.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    CompactionPerformedEvent,
    ErrorEvent,
    EventSink,
    StageCompletedEvent,
    StageStartedEvent,
)
from molexp.agent.harness.hooks import HookContext, HookPoint, HookRegistry
from molexp.agent.harness.session import Session
from molexp.agent.review import ReviewDecision
from molexp.agent.router import ModelTier

if TYPE_CHECKING:
    from pathlib import Path

    from molexp.agent.harness.execution_env import ExecResult, ExecutionEnv
    from molexp.agent.router import Router

__all__ = ["AgentHarness"]


_COMPACTION_SYSTEM_PROMPT = (
    "You compact a long agent conversation. Produce a concise summary "
    "that preserves decisions, constraints, open questions, and any "
    "facts later turns must reference. Output only the summary."
)


class AgentHarness:
    """The runtime object a mode drives — events, stages, approvals, compaction."""

    def __init__(
        self,
        *,
        session: Session,
        event_sink: EventSink,
        router: Router | None = None,
        execution_env: ExecutionEnv | None = None,
        hooks: HookRegistry | None = None,
        compaction_settings: CompactionSettings | None = None,
    ) -> None:
        self.session = session
        self._event_sink = event_sink
        self._router = router
        self._execution_env = execution_env
        self.hooks = hooks or HookRegistry()
        self.compaction_settings = compaction_settings or CompactionSettings()

    # ── injected service handles ────────────────────────────────────────────

    @property
    def router(self) -> Router:
        """The LLM dispatch :class:`~molexp.agent.router.Router`.

        Modes that drive a model call reach it here; the harness itself
        only uses it for compaction summarization.

        Raises:
            RuntimeError: if no router was supplied at construction.
        """
        if self._router is None:
            raise RuntimeError("AgentHarness has no router; AgentRunner injects one on run().")
        return self._router

    @property
    def execution_env(self) -> ExecutionEnv:
        """The injected :class:`ExecutionEnv` (subprocess + scratch dir).

        :meth:`run_subprocess` is the high-level path; a mode that needs
        to drive the env directly — e.g. AuthorMode's per-task debug loop
        running each generated test with a confined ``cwd`` — reaches it
        here.

        Raises:
            RuntimeError: if no ``execution_env`` was supplied at
                construction.
        """
        if self._execution_env is None:
            raise RuntimeError(
                "AgentHarness has no execution_env; AgentRunner supplies a "
                "LocalExecutionEnv on run()."
            )
        return self._execution_env

    # ── event emission ──────────────────────────────────────────────────────

    async def emit(self, event: AgentEvent) -> None:
        """Push one :data:`AgentEvent` to the configured sink."""
        await self._event_sink(event)

    # ── stage lifecycle ─────────────────────────────────────────────────────

    @asynccontextmanager
    async def stage(self, name: str) -> AsyncIterator[None]:
        """Bracket a unit of work with stage events + hooks.

        On entry: fires the ``before_stage`` hook, records a
        :class:`~molexp.agent.harness.session_entry.StageEntry`, emits
        :class:`~molexp.agent.harness.events.StageStartedEvent`.

        On normal exit: emits
        :class:`~molexp.agent.harness.events.StageCompletedEvent`, fires
        the ``after_stage`` hook.

        On exception: emits
        :class:`~molexp.agent.harness.events.ErrorEvent` and re-raises.
        """
        await self.hooks.dispatch(
            HookPoint.before_stage,
            HookContext(point=HookPoint.before_stage, stage_name=name),
        )
        self.session.append_stage(name)
        await self.emit(StageStartedEvent(stage_name=name))
        try:
            yield
        except Exception as exc:
            await self.emit(
                ErrorEvent(
                    message=str(exc),
                    error_type=type(exc).__name__,
                    stage_name=name,
                )
            )
            raise
        else:
            self.session.append_stage(name, completed=True)
            await self.emit(StageCompletedEvent(stage_name=name))
            await self.hooks.dispatch(
                HookPoint.after_stage,
                HookContext(point=HookPoint.after_stage, stage_name=name),
            )

    # ── unified approval gate ───────────────────────────────────────────────

    async def approve(self, gate: Any, view: Any) -> ReviewDecision:  # noqa: ANN401
        """Evaluate an approval gate — the unification point for all three.

        Replaces today's ad-hoc per-mode ``ReviewPolicy`` site. Emits
        ``approval_requested``, fires the ``before_approval`` hook (any
        handler returning a denying
        :class:`~molexp.agent.review.ReviewDecision` rejects the gate),
        records an
        :class:`~molexp.agent.harness.session_entry.ApprovalEntry`, and
        emits ``approval_decided``.

        ``gate`` is an
        :class:`~molexp.agent.modes._planning.ApprovalGate` (or any
        value with a ``.value`` string); ``view`` is any object with a
        ``summary`` attribute.
        """
        gate_name = getattr(gate, "value", str(gate))
        summary = str(getattr(view, "summary", ""))
        await self.emit(ApprovalRequestedEvent(gate=gate_name, summary=summary))

        decision = await self._evaluate_approval_hook(gate_name, summary)

        self.session.append_approval(gate_name, approved=decision.approved, reason=decision.reason)
        await self.emit(
            ApprovalDecidedEvent(
                gate=gate_name,
                approved=decision.approved,
                reason=decision.reason,
            )
        )
        return decision

    async def _evaluate_approval_hook(self, gate_name: str, summary: str) -> ReviewDecision:
        """Run ``before_approval`` handlers; the first denial wins."""
        results = await self.hooks.dispatch(
            HookPoint.before_approval,
            HookContext(
                point=HookPoint.before_approval,
                gate=gate_name,
                payload={"summary": summary},
            ),
        )
        for outcome in results:
            if isinstance(outcome, ReviewDecision) and not outcome.approved:
                return outcome
        for outcome in results:
            if isinstance(outcome, ReviewDecision):
                return outcome
        return ReviewDecision(approved=True, reason="no approval hook registered")

    # ── subprocess execution ────────────────────────────────────────────────

    async def run_subprocess(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run ``command`` through the configured :class:`ExecutionEnv`.

        Raises:
            RuntimeError: if no ``execution_env`` was supplied.
        """
        if self._execution_env is None:
            raise RuntimeError(
                "AgentHarness.run_subprocess needs an execution_env; "
                "none was supplied at construction."
            )
        return self._execution_env.exec(command, cwd=cwd, env=env, timeout=timeout)

    # ── context compaction ──────────────────────────────────────────────────

    async def compact(self) -> bool:
        """Compact the session entry tree if it exceeds the token budget.

        Picks the cut point deterministically via
        :func:`~molexp.agent.harness.compaction.prepare_compaction`,
        fires the ``before_compact`` hook (any non-``None`` result
        vetoes), summarizes the pre-cut span through the
        :class:`Router` (CHEAP tier), appends a
        :class:`~molexp.agent.harness.session_entry.CompactionEntry`,
        and emits ``compaction_performed``.

        Returns ``True`` when a compaction ran, ``False`` on a no-op
        (short conversation, vetoed, disabled).

        Raises:
            RuntimeError: if a compaction is warranted but no ``router``
                was supplied.
        """
        entries = self.session.path_to_root()
        plan = prepare_compaction(entries, self.compaction_settings)
        if plan is None:
            return False

        veto = await self.hooks.dispatch(
            HookPoint.before_compact,
            HookContext(
                point=HookPoint.before_compact,
                payload={"tokens_before": plan.tokens_before},
            ),
        )
        if veto:
            return False

        if self._router is None:
            raise RuntimeError(
                "AgentHarness.compact needs a router to summarize; "
                "none was supplied at construction."
            )

        summary = await self._summarize(plan.entries_to_summarize)
        self.session.append_compaction(
            summary=summary,
            first_kept_entry_id=plan.first_kept_entry_id,
            tokens_before=plan.tokens_before,
        )
        await self.emit(
            CompactionPerformedEvent(
                summary=summary,
                tokens_before=plan.tokens_before,
                entries_summarized=len(plan.entries_to_summarize),
            )
        )
        return True

    async def _summarize(self, entries: Sequence[Any]) -> str:
        """Render the pre-cut entries and summarize them via the router."""
        assert self._router is not None  # narrowed by the caller
        transcript = _render_transcript(entries)
        result = await self._router.complete_text(
            prompt=transcript,
            system=_COMPACTION_SYSTEM_PROMPT,
            tier=ModelTier.CHEAP,
        )
        return result.text


def _render_transcript(entries: Sequence[Any]) -> str:
    """Flatten message entries into a plain transcript for summarization."""
    from molexp.agent.harness.session_entry import MessageEntry

    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, MessageEntry):
            msg = entry.message
            lines.append(f"{msg.role}: {msg.content}")
    return "\n".join(lines)
