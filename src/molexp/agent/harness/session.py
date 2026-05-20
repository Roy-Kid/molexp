"""Cluster 2 — the ``Session`` runtime class.

:class:`Session` is the runtime conversation value the agent layer
passes through :class:`~molexp.agent.runner.AgentRunner`. It wraps a
:class:`~molexp.agent.harness.session_storage.SessionStorage` and
exposes ergonomic ``append_*`` helpers, tree navigation
(:meth:`path_to_root`, :meth:`branch`), and :meth:`build_context` — the
message-list rebuild that honours the most recent compaction cut.

It supersedes the deleted ``molexp.agent.session.AgentSession`` value
class: ``AgentRunner`` and every mode now thread a ``Session``.

Plain runtime class (it holds a live storage handle + mutable leaf
pointer), per the agent-layer "pydantic vs plain class" charter.
"""

from __future__ import annotations

import secrets

from molexp.agent.harness.session_entry import (
    ApprovalEntry,
    ArtifactEntry,
    CompactionEntry,
    MessageEntry,
    SessionEntry,
    StageEntry,
)
from molexp.agent.harness.session_storage import SessionStorage
from molexp.agent.types import Message

__all__ = ["Session"]


class Session:
    """An append-only conversation entry-tree over a :class:`SessionStorage`.

    The leaf pointer marks the active tip. Every ``append_*`` helper
    parents the new entry under the current leaf and advances the
    pointer; :meth:`branch` re-points the leaf at an interior node so
    the *next* append forks the tree non-destructively.
    """

    def __init__(
        self,
        *,
        storage: SessionStorage,
        session_id: str | None = None,
    ) -> None:
        self.storage = storage
        self.session_id = session_id or secrets.token_hex(6)

    # ── leaf pointer ────────────────────────────────────────────────────────

    @property
    def leaf_id(self) -> str | None:
        """The active tip's entry id (``None`` for an empty session)."""
        return self.storage.get_leaf_id()

    def branch(self, entry_id: str) -> None:
        """Re-point the leaf at ``entry_id`` so the next append forks.

        Raises:
            KeyError: if ``entry_id`` is not a known entry.
        """
        if self.storage.get_entry(entry_id) is None:
            raise KeyError(f"unknown entry id: {entry_id!r}")
        self.storage.set_leaf_id(entry_id)

    # ── append helpers ──────────────────────────────────────────────────────

    def append_entry(self, entry: SessionEntry) -> SessionEntry:
        """Persist a pre-built ``entry`` and advance the leaf to it."""
        self.storage.append_entry(entry)
        self.storage.set_leaf_id(entry.id)
        return entry

    def append_message(self, message: Message) -> MessageEntry:
        """Append one conversation turn under the current leaf."""
        entry = MessageEntry(
            id=self.storage.new_entry_id(),
            parent_id=self.leaf_id,
            message=message,
        )
        self.append_entry(entry)
        return entry

    def append_stage(self, stage_name: str, *, completed: bool = False) -> StageEntry:
        """Append an orchestration-stage marker under the current leaf."""
        entry = StageEntry(
            id=self.storage.new_entry_id(),
            parent_id=self.leaf_id,
            stage_name=stage_name,
            completed=completed,
        )
        self.append_entry(entry)
        return entry

    def append_artifact(self, path: str, *, description: str = "") -> ArtifactEntry:
        """Append an artefact record under the current leaf."""
        entry = ArtifactEntry(
            id=self.storage.new_entry_id(),
            parent_id=self.leaf_id,
            path=path,
            description=description,
        )
        self.append_entry(entry)
        return entry

    def append_approval(self, gate: str, *, approved: bool, reason: str = "") -> ApprovalEntry:
        """Append an approval-gate verdict under the current leaf."""
        entry = ApprovalEntry(
            id=self.storage.new_entry_id(),
            parent_id=self.leaf_id,
            gate=gate,
            approved=approved,
            reason=reason,
        )
        self.append_entry(entry)
        return entry

    def append_compaction(
        self, *, summary: str, first_kept_entry_id: str, tokens_before: int
    ) -> CompactionEntry:
        """Append a context-compaction cut under the current leaf."""
        entry = CompactionEntry(
            id=self.storage.new_entry_id(),
            parent_id=self.leaf_id,
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
        )
        self.append_entry(entry)
        return entry

    # ── navigation + context rebuild ────────────────────────────────────────

    def path_to_root(self) -> tuple[SessionEntry, ...]:
        """Return root→leaf entries on the active branch (``()`` if empty)."""
        leaf = self.leaf_id
        if leaf is None:
            return ()
        return self.storage.path_to_root(leaf)

    def build_context(self) -> tuple[Message, ...]:
        """Rebuild the conversation message list for the active branch.

        Walks the root→leaf path; if a :class:`CompactionEntry` is
        present, only the *most recent* one is honoured — entries before
        its ``first_kept_entry_id`` are dropped and replaced by a single
        ``system`` summary message. Non-message entries (stage /
        artifact / approval / model-change) never enter the context.
        """
        path = self.path_to_root()
        if not path:
            return ()

        cut = _most_recent_compaction(path)
        messages: list[Message] = []
        if cut is not None:
            messages.append(
                Message(
                    role="system",
                    content=f"[compacted context] {cut.summary}",
                )
            )
            kept = _entries_from(path, cut.first_kept_entry_id, cut.id)
        else:
            kept = path

        for entry in kept:
            if isinstance(entry, MessageEntry):
                messages.append(entry.message)
        return tuple(messages)


def _most_recent_compaction(
    path: tuple[SessionEntry, ...],
) -> CompactionEntry | None:
    """Return the last :class:`CompactionEntry` on ``path``, or ``None``."""
    for entry in reversed(path):
        if isinstance(entry, CompactionEntry):
            return entry
    return None


def _entries_from(
    path: tuple[SessionEntry, ...],
    first_kept_entry_id: str,
    compaction_entry_id: str,
) -> tuple[SessionEntry, ...]:
    """Return the post-cut slice — kept entries minus the compaction marker.

    Slices ``path`` from ``first_kept_entry_id`` to the end, dropping the
    :class:`CompactionEntry` itself (it carries no message). If
    ``first_kept_entry_id`` is absent (a stale cut pointer), degrades to
    everything after the compaction entry.
    """
    start: int | None = None
    for index, entry in enumerate(path):
        if entry.id == first_kept_entry_id:
            start = index
            break
    if start is None:
        for index, entry in enumerate(path):
            if entry.id == compaction_entry_id:
                start = index + 1
                break
    if start is None:
        return ()
    return tuple(e for e in path[start:] if e.id != compaction_entry_id)
