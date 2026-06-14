"""Context compaction.

When a session's entry tree grows past a token budget, the loop
*compacts* it: an LLM summarizes the oldest span and a
:class:`~molexp.agent.session_entry.CompactionEntry` records the
cut. This module owns the **deterministic** half — choosing the cut
point and estimating tokens — leaving the trigger and the LLM call to
:func:`molexp.agent.loops._compact.maybe_compact` (used by both
shipped loops), which routes the summary through the
:class:`~molexp.agent.router.Router` protocol (never ``pydantic_ai``
directly).

:func:`prepare_compaction` keeps the most-recent ``keep_recent_tokens``
window of entries and proposes summarizing everything older. It returns
``None`` (a no-op) when the conversation is empty, compaction is
disabled, or the recent window already covers the whole log.

Pure data + pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.session_entry import MessageEntry, SessionEntry

__all__ = [
    "CompactionPlan",
    "CompactionSettings",
    "estimate_entries_tokens",
    "estimate_tokens",
    "prepare_compaction",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

# A conservative chars-per-token ratio. Real tokenizers average ~4
# chars/token for English; rounding up keeps the estimate a safe
# over-count so we compact slightly early rather than overflow.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` with a conservative char/4 heuristic.

    Rounds up so the estimate never under-counts. Pure helper — the
    harness has no real tokenizer dependency.
    """
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


class CompactionSettings(BaseModel):
    """Tunables for the compaction policy.

    Attributes:
        enabled: When ``False``, :func:`prepare_compaction` always
            no-ops.
        keep_recent_tokens: The recent-window budget — entries summing
            to roughly this many tokens are retained verbatim.
        reserve_tokens: Head-room kept free for the next model call;
            informational here, consumed by callers deciding *when* to
            trigger compaction.
    """

    model_config = _FROZEN

    enabled: bool = True
    keep_recent_tokens: int = Field(default=8_000, gt=0)
    reserve_tokens: int = Field(default=2_000, ge=0)


class CompactionPlan(BaseModel):
    """The deterministic outcome of :func:`prepare_compaction`.

    Attributes:
        first_kept_entry_id: ``id`` of the first entry retained after
            the cut — becomes
            :attr:`~molexp.agent.session_entry.CompactionEntry.first_kept_entry_id`.
        entries_to_summarize: The pre-cut entries the LLM must
            summarize, in order.
        tokens_before: Estimated token count of the summarized span.
    """

    model_config = _FROZEN

    first_kept_entry_id: str
    entries_to_summarize: tuple[SessionEntry, ...]
    tokens_before: int


def _entry_tokens(entry: SessionEntry) -> int:
    """Estimate the token weight of one entry (messages dominate)."""
    if isinstance(entry, MessageEntry):
        return estimate_tokens(entry.message.content)
    # non-message markers are tiny; a flat small cost keeps them cheap.
    return 1


def estimate_entries_tokens(entries: tuple[SessionEntry, ...]) -> int:
    """Estimate the total token weight of ``entries``.

    Pure aggregation of the per-entry heuristic; callers use it to
    decide *when* to trigger compaction (compare against
    ``keep_recent_tokens + reserve_tokens``).
    """
    return sum(_entry_tokens(entry) for entry in entries)


def prepare_compaction(
    entries: tuple[SessionEntry, ...],
    settings: CompactionSettings,
) -> CompactionPlan | None:
    """Pick the compaction cut point keeping the recent-token window.

    Walks ``entries`` newest→oldest accumulating token estimates; every
    entry up to the ``keep_recent_tokens`` budget is retained. Everything
    older becomes :attr:`CompactionPlan.entries_to_summarize`.

    Returns ``None`` (no-op) when:

    - ``entries`` is empty;
    - ``settings.enabled`` is ``False``;
    - the retained recent window already spans every entry (nothing to
      summarize).
    """
    if not settings.enabled or not entries:
        return None

    budget = settings.keep_recent_tokens
    running = 0
    first_kept_index = 0
    for index in range(len(entries) - 1, -1, -1):
        running += _entry_tokens(entries[index])
        if running > budget:
            # This entry crosses the budget — keep it (the retained
            # window straddles the boundary) and cut everything older.
            first_kept_index = index
            break
    else:
        # never exceeded budget: the whole log is recent — nothing to do.
        return None

    if first_kept_index <= 0 or first_kept_index >= len(entries):
        # the cut would summarize nothing (or everything) — no-op.
        return None

    to_summarize = entries[:first_kept_index]
    tokens_before = sum(_entry_tokens(e) for e in to_summarize)
    return CompactionPlan(
        first_kept_entry_id=entries[first_kept_index].id,
        entries_to_summarize=to_summarize,
        tokens_before=tokens_before,
    )
