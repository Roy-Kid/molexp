"""Agent folders route ALL I/O through their injectable knowledge FileSystem.

After the OKF rehome, ``Agent`` / ``AgentSession`` are ``knowledge.Folder``
Concepts (flat layout: ``<agent>/<session>/``). This test instruments a
knowledge ``LocalFileSystem`` with a call-recording spy and asserts the agent
layer routes through ``fs`` (so a non-local backend would work) rather than
touching ``pathlib`` directly.
"""

from __future__ import annotations

import pytest

from molexp.agent.folders import Agent
from molexp.knowledge import LocalFileSystem


class _SpyFileSystem:
    """Wraps knowledge ``LocalFileSystem``, recording every method call."""

    def __init__(self) -> None:
        self._real = LocalFileSystem()
        self.calls: list[tuple[str, str]] = []

    def _record(self, name: str, path: object) -> None:
        self.calls.append((name, str(path)))

    def __getattr__(self, name: str):
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def wrapped(*args: object, **kwargs: object) -> object:
            self._record(name, args[0] if args else kwargs.get("path", ""))
            return attr(*args, **kwargs)

        return wrapped


@pytest.fixture
def spy_agent(tmp_path):
    """An Agent rooted at a tmp bundle, backed by a recording spy fs."""
    fs = _SpyFileSystem()
    agent = Agent(name="reviewer", root=tmp_path / "lab", fs=fs)
    return agent, fs


def _ops_for_path(calls: list[tuple[str, str]], rel: str) -> set[str]:
    return {op for op, path in calls if rel in path}


def test_agent_materialize_routes_through_fs(spy_agent):
    agent, fs = spy_agent
    agent.materialize()
    ops = _ops_for_path(fs.calls, "reviewer")
    assert "mkdir" in ops, f"materialize must mkdir via fs; ops were {ops!r}"
    assert "write_text" in ops, f"materialize must write meta.yaml via fs; ops were {ops!r}"


def test_session_messages_path_is_flat(spy_agent):
    agent, _fs = spy_agent
    session = agent.add_session("chat-1")
    # flat layout: <agent>/<session>/messages.jsonl (no agents/ + agent_sessions/)
    assert str(session.messages_path).endswith("reviewer/chat-1/messages.jsonl")


def test_read_messages_routes_through_fs(spy_agent):
    agent, fs = spy_agent
    session = agent.add_session("chat-1")
    fs.calls.clear()
    assert session.read_messages() == ()
    ops = _ops_for_path(fs.calls, "messages.jsonl")
    assert "exists" in ops, f"read_messages must call fs.exists; ops were {ops!r}"


def test_write_messages_empty_removes_via_fs(spy_agent):
    agent, fs = spy_agent
    session = agent.add_session("chat-1")
    fs.write_bytes(session.messages_path, b"{}\n")
    fs.calls.clear()
    session.write_messages(())  # empty → remove via fs
    ops = _ops_for_path(fs.calls, "messages.jsonl")
    assert "remove" in ops, f"write_messages(empty) must remove via fs.remove; ops were {ops!r}"


def test_sessions_round_trip_via_registry(tmp_path):
    fs = _SpyFileSystem()
    agent = Agent(name="reviewer", root=tmp_path / "lab", fs=fs)
    agent.add_session("chat-1", goal_summary="solve X", status="running")

    # a fresh Agent handle (empty child cache) reconstructs sessions from disk
    fresh_fs = _SpyFileSystem()
    reloaded = Agent(name="reviewer", root=tmp_path / "lab", fs=fresh_fs)
    assert [s.name for s in reloaded.list_sessions()] == ["chat-1"]
    assert isinstance(reloaded.get_session("chat-1"), type(agent.get_session("chat-1")))

    # disk is the source of truth — a reconstructed handle sees persisted state,
    # not constructor defaults (read-through, no stale in-memory shadow)
    session = reloaded.get_session("chat-1")
    assert session.goal_summary == "solve X"
    assert session.status == "running"
    assert session.read_session_meta().goal_summary == "solve X"
