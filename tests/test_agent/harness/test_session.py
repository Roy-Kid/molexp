"""``Session`` entry-tree behaviour (spec ac-003, ac-004)."""

from __future__ import annotations

import pytest

from molexp.agent.session import Session
from molexp.agent.session_entry import CompactionEntry, MessageEntry
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import Message


def _new_session() -> Session:
    return Session(storage=InMemorySessionStorage(), session_id="s1")


def test_session_id_is_preserved() -> None:
    sess = Session(storage=InMemorySessionStorage(), session_id="abc")
    assert sess.session_id == "abc"


def test_session_assigns_id_when_unspecified() -> None:
    a = Session(storage=InMemorySessionStorage())
    b = Session(storage=InMemorySessionStorage())
    assert a.session_id != b.session_id
    assert len(a.session_id) >= 6


def test_append_message_advances_leaf() -> None:
    sess = _new_session()
    e1 = sess.append_message(Message(role="user", content="hi"))
    assert sess.leaf_id == e1.id
    e2 = sess.append_message(Message(role="assistant", content="hello"))
    assert sess.leaf_id == e2.id
    assert e2.parent_id == e1.id


def test_append_stage_and_artifact_and_approval() -> None:
    sess = _new_session()
    sess.append_message(Message(role="user", content="go"))
    stage = sess.append_stage("draft")
    artifact = sess.append_artifact("out.txt", description="result")
    approval = sess.append_approval("approve_direction", approved=True)
    assert stage.stage_name == "draft"
    assert artifact.path == "out.txt"
    assert approval.approved is True
    # all chained under the leaf
    assert sess.leaf_id == approval.id


def test_path_to_root_reproduces_conversation() -> None:
    sess = _new_session()
    sess.append_message(Message(role="user", content="q1"))
    sess.append_message(Message(role="assistant", content="a1"))
    sess.append_message(Message(role="user", content="q2"))
    path = sess.path_to_root()
    contents = [e.message.content for e in path if isinstance(e, MessageEntry)]
    assert contents == ["q1", "a1", "q2"]


def test_branch_is_non_destructive() -> None:
    """Appending under an interior parent leaves the original branch reachable."""
    sess = _new_session()
    root = sess.append_message(Message(role="user", content="root"))
    main_a = sess.append_message(Message(role="assistant", content="main-a"))
    main_b = sess.append_message(Message(role="user", content="main-b"))

    original_tip = main_b.id
    original_path = [e.id for e in sess.storage.path_to_root(original_tip)]

    # Fork from an interior node (root): a brand-new branch.
    sess.branch(root.id)
    fork_entry = sess.append_message(Message(role="assistant", content="fork"))

    # The original branch is still fully reachable + unchanged.
    assert [e.id for e in sess.storage.path_to_root(original_tip)] == original_path
    # The new leaf points at the fork.
    assert sess.leaf_id == fork_entry.id
    assert fork_entry.parent_id == root.id
    # main-a / main-b never appear on the fork's path.
    fork_ids = {e.id for e in sess.storage.path_to_root(fork_entry.id)}
    assert main_a.id not in fork_ids
    assert main_b.id not in fork_ids


def test_build_context_returns_message_list() -> None:
    sess = _new_session()
    sess.append_message(Message(role="user", content="hi"))
    sess.append_stage("noise")  # non-message entries are skipped
    sess.append_message(Message(role="assistant", content="yo"))
    msgs = sess.build_context()
    assert [m.content for m in msgs] == ["hi", "yo"]
    assert all(isinstance(m, Message) for m in msgs)


def test_build_context_honors_compaction_cut() -> None:
    """A CompactionEntry mid-log drops pre-cut entries and prepends the summary."""
    storage = InMemorySessionStorage()
    sess = Session(storage=storage, session_id="c")
    sess.append_message(Message(role="user", content="old-1"))
    sess.append_message(Message(role="assistant", content="old-2"))
    keep = sess.append_message(Message(role="user", content="kept-q"))
    # Insert a compaction cut whose first_kept_entry_id points at `keep`.
    cut = CompactionEntry(
        id=storage.new_entry_id(),
        parent_id=sess.leaf_id,
        summary="summary of old turns",
        first_kept_entry_id=keep.id,
        tokens_before=42,
    )
    sess.append_entry(cut)
    sess.append_message(Message(role="assistant", content="kept-a"))

    msgs = sess.build_context()
    # Pre-cut "old-1" / "old-2" are gone; summary leads; kept turns follow.
    assert msgs[0].role == "system"
    assert "summary of old turns" in msgs[0].content
    contents = [m.content for m in msgs[1:]]
    assert contents == ["kept-q", "kept-a"]
    assert "old-1" not in str(msgs)


def test_build_context_empty_session_is_empty() -> None:
    sess = _new_session()
    assert sess.build_context() == ()


def test_branch_to_unknown_entry_raises() -> None:
    sess = _new_session()
    with pytest.raises(KeyError):
        sess.branch("ghost")
