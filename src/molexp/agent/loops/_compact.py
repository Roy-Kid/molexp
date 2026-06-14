"""Compaction trigger shared by the shipped loops.

:func:`maybe_compact` is the impure half of context compaction — the
deterministic cut-point selection lives in
:mod:`molexp.agent.compaction` (pure data + pure functions). Both
shipped loops (:class:`~molexp.agent.loops.chat.ChatLoop`,
:class:`~molexp.agent.loops.interactive.InteractiveLoop`) call it at
the one natural seam they share: right after appending the user
message, right before the model call.

The trigger is conservative: nothing happens until the *effective*
context (entries after the most recent compaction cut) is estimated
above ``keep_recent_tokens + reserve_tokens``. When it fires, the
oldest span is summarized through :meth:`Router.complete_text` on the
``CHEAP`` tier, a :class:`~molexp.agent.session_entry.CompactionEntry`
records the cut, and a
:class:`~molexp.agent.events.CompactionPerformedEvent` flows to the
sink. Opt out per loop via
``ChatLoopConfig(compaction=CompactionSettings(enabled=False))`` (same
for ``InteractiveLoopConfig``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mollog import get_logger

from molexp.agent.compaction import (
    CompactionSettings,
    estimate_entries_tokens,
    prepare_compaction,
)
from molexp.agent.events import AsyncIteratorEventSink, CompactionPerformedEvent
from molexp.agent.router import ModelTier
from molexp.agent.session import _entries_from, _most_recent_compaction
from molexp.agent.session_entry import MessageEntry, SessionEntry

if TYPE_CHECKING:
    from molexp.agent.runtime import AgentRuntime

_LOG = get_logger(__name__)

__all__ = ["maybe_compact"]

# System prompt for the compaction-summary LLM call. Kept deliberately
# terse: the summary stands in for the dropped span inside future
# contexts, so it must be self-contained and dense.
_SUMMARY_SYSTEM_PROMPT = (
    "You compact an agent conversation. Summarize the transcript below "
    "into a concise brief that preserves facts, decisions, open "
    "questions, and tool results. Output only the summary."
)


def _effective_entries(
    path: tuple[SessionEntry, ...],
) -> tuple[tuple[SessionEntry, ...], str | None]:
    """Return the post-cut entry slice and the prior summary (if any).

    A session that already compacted still carries every entry on the
    root→leaf path; only entries *after* the most recent cut count
    toward the next trigger, and the prior summary is folded into the
    next one so no information silently drops.
    """
    cut = _most_recent_compaction(path)
    if cut is None:
        return path, None
    return _entries_from(path, cut.first_kept_entry_id, cut.id), cut.summary


def _render_transcript(
    entries: tuple[SessionEntry, ...],
    prior_summary: str | None,
) -> str:
    """Render the to-be-summarized span as a plain-text transcript."""
    lines: list[str] = []
    if prior_summary is not None:
        lines.append(f"[earlier summary] {prior_summary}")
    for entry in entries:
        if isinstance(entry, MessageEntry):
            lines.append(f"{entry.message.role}: {entry.message.content}")
    return "\n".join(lines)


async def maybe_compact(
    *,
    runtime: AgentRuntime,
    sink: AsyncIteratorEventSink,
    settings: CompactionSettings,
    loop_name: str,
) -> bool:
    """Compact the session entry tree if it exceeds the token budget.

    Returns ``True`` when a compaction was performed, ``False`` when
    the context was left untouched (disabled, under budget, or no
    viable cut point).
    """
    if not settings.enabled:
        return False

    path = runtime.session.path_to_root()
    effective, prior_summary = _effective_entries(path)
    trigger_tokens = settings.keep_recent_tokens + settings.reserve_tokens
    estimated = estimate_entries_tokens(effective)
    if estimated <= trigger_tokens:
        return False

    plan = prepare_compaction(effective, settings)
    if plan is None:
        return False

    transcript = _render_transcript(plan.entries_to_summarize, prior_summary)
    result = await runtime.router.complete_text(
        prompt=transcript,
        system=_SUMMARY_SYSTEM_PROMPT,
        tier=ModelTier.CHEAP,
    )
    runtime.session.append_compaction(
        summary=result.text,
        first_kept_entry_id=plan.first_kept_entry_id,
        tokens_before=plan.tokens_before,
    )
    await sink(
        CompactionPerformedEvent(
            summary=result.text,
            tokens_before=plan.tokens_before,
            entries_summarized=len(plan.entries_to_summarize),
        )
    )
    _LOG.info(
        f"[{loop_name}] compacted session — estimated {estimated} tokens "
        f"exceeded trigger {trigger_tokens}; summarized "
        f"{len(plan.entries_to_summarize)} entries ({plan.tokens_before} tokens)"
    )
    return True
