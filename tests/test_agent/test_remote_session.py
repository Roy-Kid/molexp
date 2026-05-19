"""Acceptance test for PR 4: agent layer offloads "local-only" assumption.

Verifies that :class:`molexp.agent.folders.Agent` and
:class:`molexp.agent.folders.AgentSession` route ALL I/O through
``self._fs`` and therefore would work against a non-local
:class:`FileSystem` implementation (e.g. ``RemoteFileSystem``).  Before
PR 4 these subclasses bound directly to :meth:`pathlib.Path.exists` /
``.mkdir`` / ``.read_bytes`` and silently broke when mounted under a
non-:class:`LocalFileSystem` workspace.

This test instruments :class:`LocalFileSystem` with a spy that records
every method call, then asserts that all of the agent-layer I/O calls
went through fs.  A real :class:`RemoteFileSystem` would route the same
calls through SSH (subject to a richer molq Transport implementation, a
separate workstream).
"""

from __future__ import annotations

import pytest

from molexp.agent.folders import Agent, AgentSession
from molexp.workspace import Workspace
from molexp.workspace.fs_local import LocalFileSystem


class _SpyFileSystem:
    """Wraps :class:`LocalFileSystem` and records every method call.

    Behaves identically to LocalFileSystem for I/O but exposes
    ``.calls`` — the list of ``(method_name, path)`` tuples — so tests
    can assert the agent layer routed through fs instead of touching
    pathlib directly.
    """

    def __init__(self) -> None:
        self._real = LocalFileSystem()
        self.calls: list[tuple[str, str]] = []

    def _record(self, name: str, path: object) -> None:
        try:
            self.calls.append((name, str(path)))
        except Exception:
            self.calls.append((name, "<unrepresentable>"))

    def __getattr__(self, name: str):  # noqa: ANN204
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):  # noqa: ANN202
            if args:
                self._record(name, args[0])
            else:
                self._record(name, kwargs.get("path", ""))
            return attr(*args, **kwargs)

        return wrapped


@pytest.fixture
def spy_workspace(tmp_path):
    """Workspace whose FileSystem is a recording spy over LocalFileSystem."""
    fs = _SpyFileSystem()
    ws = Workspace(root=tmp_path / "lab", fs=fs)
    ws.materialize()
    fs.calls.clear()  # ignore the materialize call from this point on
    return ws, fs


def _ops_for_path(calls: list[tuple[str, str]], rel: str) -> set[str]:
    """Set of fs-method names that touched any path containing ``rel``."""
    return {op for op, path in calls if rel in path}


def test_agent_mount_routes_through_fs(spy_workspace):
    """``ws.add_folder(Agent(...))`` should mkdir + write metadata via fs."""
    ws, fs = spy_workspace
    ws.add_folder(Agent(name="reviewer"))

    ops = _ops_for_path(fs.calls, "agents/reviewer")
    assert "mkdir" in ops, (
        f"Agent mount must mkdir via fs (not pathlib.Path.mkdir); ops were {ops!r}"
    )
    # atomic_write_json is how _save_metadata persists agent.json
    assert any("write" in op or "atomic" in op for op in ops), (
        f"Agent mount must write metadata via fs; ops were {ops!r}"
    )


def test_agent_session_messages_path_uses_fs_join(spy_workspace):
    """``session.messages_path`` should be a fs.join'd path under the session dir."""
    ws, fs = spy_workspace
    agent = ws.add_folder(Agent(name="reviewer"))
    session = agent.add_session("chat-1")

    expected_tail = "agents/reviewer/agent_sessions/chat-1/messages.jsonl"
    assert str(session.messages_path).endswith(expected_tail), (
        f"expected messages_path to end with {expected_tail!r}; got {session.messages_path}"
    )


def test_agent_session_read_messages_routes_through_fs(spy_workspace):
    """``read_messages()`` on an empty session must call ``fs.exists``, not pathlib."""
    ws, fs = spy_workspace
    agent = ws.add_folder(Agent(name="reviewer"))
    session = agent.add_session("chat-1")
    fs.calls.clear()

    result = session.read_messages()
    assert result == ()

    ops = _ops_for_path(fs.calls, "messages.jsonl")
    assert "exists" in ops, (
        f"read_messages must call fs.exists (not pathlib.Path.exists); ops were {ops!r}"
    )


def test_agent_session_write_messages_routes_through_fs(spy_workspace):
    """``write_messages([])`` removes via fs.remove + fs.exists, not pathlib."""
    ws, fs = spy_workspace
    agent = ws.add_folder(Agent(name="reviewer"))
    session = agent.add_session("chat-1")

    # Plant a fake messages.jsonl so the empty-write branch hits .remove
    fs.write_bytes(session.messages_path, b"{}\n")
    fs.calls.clear()

    session.write_messages(())  # empty → should remove

    ops = _ops_for_path(fs.calls, "messages.jsonl")
    assert "exists" in ops
    assert "remove" in ops, (
        f"write_messages(empty) must remove via fs.remove; ops were {ops!r}"
    )


def test_agent_session_reload_routes_through_fs(spy_workspace, tmp_path):
    """``from_disk`` (via get_folder on a fresh Workspace) must use fs."""
    ws, _fs = spy_workspace
    agent = ws.add_folder(Agent(name="reviewer"))
    agent.add_session("chat-1")

    # New workspace handle → empty children cache → forces from_disk path.
    fresh_fs = _SpyFileSystem()
    fresh_ws = Workspace(root=tmp_path / "lab", fs=fresh_fs)

    reloaded = fresh_ws.get_folder("reviewer", cls=Agent)
    sessions = reloaded.list_sessions()
    assert [s.name for s in sessions] == ["chat-1"]

    # Reload must have opened agent.json + agent_session.json via fs
    agent_meta_ops = _ops_for_path(fresh_fs.calls, "agents/reviewer/agent.json")
    session_meta_ops = _ops_for_path(
        fresh_fs.calls, "agent_sessions/chat-1/agent_session.json"
    )
    assert agent_meta_ops, "from_disk(Agent) must touch agent.json via fs"
    assert session_meta_ops, "from_disk(AgentSession) must touch agent_session.json via fs"
