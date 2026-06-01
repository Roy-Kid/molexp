"""The ``SessionStorage`` repository + two implementations.

:class:`SessionStorage` is a Repository-pattern :class:`typing.Protocol`
the :class:`~molexp.agent.session.Session` runtime class talks
to. It abstracts *where* the append-only entry tree lives so the same
``Session`` code drives an on-disk JSONL file (production) or an
in-memory dict (tests).

Two concrete implementations ship here:

- :class:`JsonlSessionStorage` — anchors to a directory (a
  :class:`~molexp.agent.folders.AgentSession` ``Folder`` dir in
  production), writing an append-only ``entries.jsonl`` plus a ``leaf``
  pointer file.
- :class:`InMemorySessionStorage` — a process-local dict; the default
  for unit tests and workspace-less runs.

Both are plain runtime classes (they hold mutable state / do I/O), per
the agent-layer "pydantic vs plain class" charter.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import TypeAdapter

from molexp.agent.session_entry import SessionEntry

__all__ = [
    "InMemorySessionStorage",
    "JsonlSessionStorage",
    "SessionStorage",
]

_ENTRY_ADAPTER: TypeAdapter[SessionEntry] = TypeAdapter(SessionEntry)


@runtime_checkable
class SessionStorage(Protocol):
    """Append-only repository for the session entry tree.

    Implementations are non-destructive: :meth:`append_entry` only ever
    adds; an entry whose ``parent_id`` points at an interior node forks
    the tree. The ``leaf`` pointer marks the active tip.
    """

    def new_entry_id(self) -> str:
        """Return a fresh, collision-resistant entry id."""
        ...

    def append_entry(self, entry: SessionEntry) -> None:
        """Persist ``entry``. Never overwrites an existing entry."""
        ...

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        """Return the entry with ``id == entry_id``, or ``None``."""
        ...

    def path_to_root(self, leaf_id: str) -> tuple[SessionEntry, ...]:
        """Return root→``leaf_id`` entries in order; ``()`` if unknown."""
        ...

    def get_leaf_id(self) -> str | None:
        """Return the active tip's entry id, or ``None`` if unset."""
        ...

    def set_leaf_id(self, leaf_id: str) -> None:
        """Mark ``leaf_id`` as the active tip."""
        ...


def _path_to_root(leaf_id: str, lookup: dict[str, SessionEntry]) -> tuple[SessionEntry, ...]:
    """Walk ``parent_id`` pointers root-ward; shared by both impls."""
    chain: list[SessionEntry] = []
    cursor: str | None = leaf_id
    seen: set[str] = set()
    while cursor is not None:
        if cursor in seen:  # defensive — a corrupt log could cycle
            break
        entry = lookup.get(cursor)
        if entry is None:
            return ()
        seen.add(cursor)
        chain.append(entry)
        cursor = entry.parent_id
    chain.reverse()
    return tuple(chain)


class InMemorySessionStorage:
    """Process-local :class:`SessionStorage` — the test / workspace-less impl."""

    def __init__(self) -> None:
        self._entries: dict[str, SessionEntry] = {}
        self._leaf_id: str | None = None

    def new_entry_id(self) -> str:
        return secrets.token_hex(8)

    def append_entry(self, entry: SessionEntry) -> None:
        if entry.id in self._entries:
            raise ValueError(f"entry id {entry.id!r} already exists")
        self._entries[entry.id] = entry

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._entries.get(entry_id)

    def path_to_root(self, leaf_id: str) -> tuple[SessionEntry, ...]:
        return _path_to_root(leaf_id, self._entries)

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def set_leaf_id(self, leaf_id: str) -> None:
        self._leaf_id = leaf_id


_ENTRIES_FILENAME = "entries.jsonl"
_LEAF_FILENAME = "leaf"


class JsonlSessionStorage:
    """On-disk :class:`SessionStorage` — append-only ``entries.jsonl``.

    Anchors to a directory: in production the
    :class:`~molexp.agent.folders.AgentSession` ``Folder`` dir, so the
    session tree sits next to the folder's ``agent_session.json``.
    The directory is created lazily on first write.

    Entries are appended one JSON object per line; the directory's
    ``leaf`` file holds the active tip id. A fresh instance over an
    existing directory re-reads both — resume is free.
    """

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._entries: dict[str, SessionEntry] = {}
        self._order: list[str] = []
        self._leaf_id: str | None = None
        self._loaded = False

    # ── lazy load ───────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        entries_path = self._dir / _ENTRIES_FILENAME
        if entries_path.exists():
            for line in entries_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                entry = _ENTRY_ADAPTER.validate_json(stripped)
                self._entries[entry.id] = entry
                self._order.append(entry.id)
        leaf_path = self._dir / _LEAF_FILENAME
        if leaf_path.exists():
            text = leaf_path.read_text(encoding="utf-8").strip()
            self._leaf_id = text or None

    # ── SessionStorage protocol ─────────────────────────────────────────────

    def new_entry_id(self) -> str:
        return secrets.token_hex(8)

    def append_entry(self, entry: SessionEntry) -> None:
        self._ensure_loaded()
        if entry.id in self._entries:
            raise ValueError(f"entry id {entry.id!r} already exists")
        self._dir.mkdir(parents=True, exist_ok=True)
        line = _ENTRY_ADAPTER.dump_json(entry).decode("utf-8")
        with (self._dir / _ENTRIES_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._entries[entry.id] = entry
        self._order.append(entry.id)

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        self._ensure_loaded()
        return self._entries.get(entry_id)

    def path_to_root(self, leaf_id: str) -> tuple[SessionEntry, ...]:
        self._ensure_loaded()
        return _path_to_root(leaf_id, self._entries)

    def get_leaf_id(self) -> str | None:
        self._ensure_loaded()
        return self._leaf_id

    def set_leaf_id(self, leaf_id: str) -> None:
        self._ensure_loaded()
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / (_LEAF_FILENAME + ".tmp")
        tmp.write_text(leaf_id, encoding="utf-8")
        tmp.replace(self._dir / _LEAF_FILENAME)
        self._leaf_id = leaf_id
