"""Context-compaction tests (spec ac-005, ac-006)."""

from __future__ import annotations

from molexp.agent.compaction import (
    CompactionPlan,
    CompactionSettings,
    estimate_tokens,
    prepare_compaction,
)
from molexp.agent.session_entry import MessageEntry
from molexp.agent.types import Message


def _msg_entry(idx: int, content: str, parent: str | None) -> MessageEntry:
    return MessageEntry(
        id=f"e{idx}",
        parent_id=parent,
        message=Message(role="user", content=content),
    )


def _chain(contents: list[str]) -> tuple[MessageEntry, ...]:
    entries: list[MessageEntry] = []
    parent: str | None = None
    for idx, content in enumerate(contents):
        entry = _msg_entry(idx, content, parent)
        entries.append(entry)
        parent = entry.id
    return tuple(entries)


# ── token estimation ───────────────────────────────────────────────────────


def test_estimate_tokens_is_char_over_four() -> None:
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 0


def test_estimate_tokens_rounds_up() -> None:
    # 5 chars / 4 -> 2 (conservative, rounds up)
    assert estimate_tokens("abcde") == 2


# ── prepare_compaction no-ops ──────────────────────────────────────────────


def test_prepare_compaction_empty_list_is_noop() -> None:
    assert prepare_compaction((), CompactionSettings()) is None


def test_prepare_compaction_disabled_is_noop() -> None:
    entries = _chain(["x" * 1000] * 10)
    settings = CompactionSettings(enabled=False, keep_recent_tokens=10)
    assert prepare_compaction(entries, settings) is None


def test_prepare_compaction_under_budget_is_noop() -> None:
    """Small conversations need no cut."""
    entries = _chain(["short"] * 3)
    settings = CompactionSettings(keep_recent_tokens=10_000)
    assert prepare_compaction(entries, settings) is None


def test_prepare_compaction_already_compacted_first_entry_is_noop() -> None:
    """If only the most-recent window remains, there is nothing to cut."""
    # one big entry — keeping the recent window already covers everything.
    entries = _chain(["x" * 4000])
    settings = CompactionSettings(keep_recent_tokens=100)
    assert prepare_compaction(entries, settings) is None


# ── prepare_compaction cut-point selection ─────────────────────────────────


def test_prepare_compaction_keeps_recent_token_window() -> None:
    # 6 entries, 100 tokens each (400 chars). keep_recent_tokens=250
    # -> keep the last 3 entries (300 tokens), summarize the first 3.
    entries = _chain(["x" * 400] * 6)
    settings = CompactionSettings(keep_recent_tokens=250)
    plan = prepare_compaction(entries, settings)
    assert isinstance(plan, CompactionPlan)
    assert plan.first_kept_entry_id == entries[3].id
    summarized_ids = {e.id for e in plan.entries_to_summarize}
    assert summarized_ids == {entries[0].id, entries[1].id, entries[2].id}
    assert plan.tokens_before == sum(estimate_tokens("x" * 400) for _ in range(3))


def test_prepare_compaction_retained_window_approximates_budget() -> None:
    entries = _chain(["x" * 400] * 6)
    settings = CompactionSettings(keep_recent_tokens=250)
    plan = prepare_compaction(entries, settings)
    assert plan is not None
    kept = entries[entries.index(_find(entries, plan.first_kept_entry_id)) :]
    kept_tokens = sum(estimate_tokens("x" * 400) for _ in kept)
    # retained window stays within one entry of the budget
    assert kept_tokens <= settings.keep_recent_tokens + estimate_tokens("x" * 400)


def _find(entries: tuple[MessageEntry, ...], entry_id: str) -> MessageEntry:
    for entry in entries:
        if entry.id == entry_id:
            return entry
    raise AssertionError(entry_id)


# ── CompactionSettings ─────────────────────────────────────────────────────


def test_compaction_settings_defaults() -> None:
    settings = CompactionSettings()
    assert settings.enabled is True
    assert settings.keep_recent_tokens > 0
    assert settings.reserve_tokens >= 0


def test_compaction_settings_is_frozen() -> None:
    import pytest
    from pydantic import ValidationError

    settings = CompactionSettings()
    with pytest.raises(ValidationError):
        settings.enabled = False  # type: ignore[misc]
