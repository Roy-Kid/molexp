"""Agent Concept folders route I/O through their injectable workspace fs.

After the OKF rehome, ``Agent`` / ``AgentSession`` are
``molexp.workspace.Folder`` Concepts. This locks the backend-agnostic
invariant: their overrides go through the injected ``FileSystem``
(so a non-local backend works) rather than touching ``pathlib`` directly.
The observable outcomes of these methods are owned by ``test_folders.py``;
here we assert only the fs-routing.
"""

from __future__ import annotations

import pytest

from molexp.agent.folders import Agent
from molexp.workspace.fs_local import LocalFileSystem


class _SpyFileSystem:
    """Wraps workspace ``LocalFileSystem``, recording every method call."""

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
    agent = Agent(name="reviewer", root_path=tmp_path / "lab", fs=fs)
    return agent, fs


class TestAgent:
    def test_materialize_routes_io_through_injectable_fs(self, spy_agent) -> None:
        agent, fs = spy_agent
        agent.materialize()
        ops = {op for op, path in fs.calls if "reviewer" in path}
        assert "mkdir" in ops, f"materialize must mkdir via fs; ops were {ops!r}"
        assert "atomic_write_text" in ops, (
            f"materialize must write meta.json via fs; ops were {ops!r}"
        )
