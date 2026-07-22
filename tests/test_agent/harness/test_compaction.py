"""Tests for ``molexp.agent.compaction`` — cut-point selection + token estimate."""

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


class TestEstimateTokens:
    def test_estimate_tokens_rounds_char_count_up_by_four(self) -> None:
        assert estimate_tokens("a" * 40) == 10
        assert estimate_tokens("") == 0


class TestPrepareCompaction:
    def test_disabled_settings_is_a_noop(self) -> None:
        entries = _chain(["x" * 1000] * 10)
        settings = CompactionSettings(enabled=False, keep_recent_tokens=10)
        assert prepare_compaction(entries, settings) is None

    def test_conversation_under_budget_is_a_noop(self) -> None:
        entries = _chain(["short"] * 3)
        settings = CompactionSettings(keep_recent_tokens=10_000)
        assert prepare_compaction(entries, settings) is None

    def test_cut_point_keeps_recent_token_window(self) -> None:
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
