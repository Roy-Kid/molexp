"""Tests for ``molexp.agent.session_storage`` — protocol parity + jsonl persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.session_entry import MessageEntry, StageEntry
from molexp.agent.session_storage import (
    InMemorySessionStorage,
    JsonlSessionStorage,
    SessionStorage,
)
from molexp.agent.types import Message


@pytest.fixture(params=["in_memory", "jsonl"])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> SessionStorage:
    if request.param == "in_memory":
        return InMemorySessionStorage()
    return JsonlSessionStorage(tmp_path / "sess")


def _msg_entry(storage: SessionStorage, parent_id: str | None) -> MessageEntry:
    return MessageEntry(
        id=storage.new_entry_id(),
        parent_id=parent_id,
        message=Message(role="user", content="hello"),
    )


class TestSessionStorage:
    """Parity contract every ``SessionStorage`` implementation must satisfy."""

    def test_append_then_get_round_trips(self, storage: SessionStorage) -> None:
        entry = _msg_entry(storage, None)
        storage.append_entry(entry)
        assert storage.get_entry(entry.id) == entry

    def test_get_missing_entry_returns_none(self, storage: SessionStorage) -> None:
        assert storage.get_entry("nope") is None

    def test_leaf_pointer_starts_unset_then_round_trips(self, storage: SessionStorage) -> None:
        assert storage.get_leaf_id() is None
        entry = _msg_entry(storage, None)
        storage.append_entry(entry)
        storage.set_leaf_id(entry.id)
        assert storage.get_leaf_id() == entry.id

    def test_path_to_root_walks_the_parent_chain_in_order(self, storage: SessionStorage) -> None:
        e1 = _msg_entry(storage, None)
        storage.append_entry(e1)
        e2 = MessageEntry(
            id=storage.new_entry_id(),
            parent_id=e1.id,
            message=Message(role="assistant", content="a"),
        )
        storage.append_entry(e2)
        e3 = StageEntry(id=storage.new_entry_id(), parent_id=e2.id, stage_name="draft")
        storage.append_entry(e3)
        path = storage.path_to_root(e3.id)
        assert [e.id for e in path] == [e1.id, e2.id, e3.id]


class TestJsonlSessionStorage:
    def test_persists_entries_and_leaf_across_instances(self, tmp_path: Path) -> None:
        """A fresh instance over the same dir re-reads prior entries + leaf."""
        root = tmp_path / "persist"
        first = JsonlSessionStorage(root)
        e1 = _msg_entry(first, None)
        first.append_entry(e1)
        first.set_leaf_id(e1.id)

        second = JsonlSessionStorage(root)
        assert second.get_entry(e1.id) == e1
        assert second.get_leaf_id() == e1.id
