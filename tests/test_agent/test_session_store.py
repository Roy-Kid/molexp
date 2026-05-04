"""Phase 1a: SessionStore atomic JSON + JSONL writes."""

from __future__ import annotations

from pathlib import Path

from molexp.agent import Goal, Message, SessionStatus
from molexp.agent.state import SessionMetadata, SessionStore


def test_metadata_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    meta = SessionMetadata(
        session_id="abc",
        goal=Goal(description="explore"),
        status=SessionStatus.RUNNING,
    )
    store.write_metadata(meta)
    loaded = store.read_metadata("abc")
    assert loaded is not None
    assert loaded.session_id == "abc"
    assert loaded.status is SessionStatus.RUNNING
    assert loaded.goal.description == "explore"


def test_messages_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_messages(
        "abc",
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ],
    )
    out = store.read_messages("abc")
    assert [m.role for m in out] == ["user", "assistant"]
    assert [m.content for m in out] == ["hi", "hello"]


def test_list_sessions_returns_sorted(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    for sid in ["bbb", "aaa", "ccc"]:
        store.write_metadata(
            SessionMetadata(
                session_id=sid,
                goal=Goal(description=sid),
                status=SessionStatus.RUNNING,
            )
        )
    listed = store.list_sessions()
    assert [m.session_id for m in listed] == ["aaa", "bbb", "ccc"]


def test_metadata_write_is_atomic(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    meta = SessionMetadata(
        session_id="abc",
        goal=Goal(description="explore"),
        status=SessionStatus.RUNNING,
    )
    store.write_metadata(meta)
    # No leftover .tmp files after a successful write.
    leftovers = list((tmp_path / "abc").glob("*.tmp"))
    assert leftovers == []
