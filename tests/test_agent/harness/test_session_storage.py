"""``SessionStorage`` protocol + two-implementation parity (spec ac-002)."""

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


def test_append_then_get_round_trips(storage: SessionStorage) -> None:
    entry = _msg_entry(storage, None)
    storage.append_entry(entry)
    loaded = storage.get_entry(entry.id)
    assert loaded == entry


def test_get_missing_entry_returns_none(storage: SessionStorage) -> None:
    assert storage.get_entry("nope") is None


def test_leaf_pointer_round_trips(storage: SessionStorage) -> None:
    assert storage.get_leaf_id() is None
    entry = _msg_entry(storage, None)
    storage.append_entry(entry)
    storage.set_leaf_id(entry.id)
    assert storage.get_leaf_id() == entry.id


def test_path_to_root_walks_parent_chain(storage: SessionStorage) -> None:
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


def test_jsonl_storage_persists_across_instances(tmp_path: Path) -> None:
    """A new ``JsonlSessionStorage`` over the same dir sees prior entries."""
    root = tmp_path / "persist"
    first = JsonlSessionStorage(root)
    e1 = _msg_entry(first, None)
    first.append_entry(e1)
    first.set_leaf_id(e1.id)

    second = JsonlSessionStorage(root)
    assert second.get_entry(e1.id) == e1
    assert second.get_leaf_id() == e1.id
