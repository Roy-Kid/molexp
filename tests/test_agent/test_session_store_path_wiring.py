"""Tests for agent-side store wiring through ``Workspace.subsystem_store`` (spec ac-009).

Each surviving store (sessions, mcp) must materialize its on-disk state under
the corresponding subsystem directory:

- ``<root>/.subsystems/agent.sessions/<id>/...``
- ``<root>/.subsystems/agent.mcp/mcp.json``

The previously-covered ``skills`` and ``tools`` wirings were removed when
``agent-pydanticai-rectification`` deleted those parallel-to-pydantic-ai
subpackages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path)
    ws.materialize()
    return ws


class TestSessionStoreWiring:
    def test_session_store_writes_under_subsystem_dir(self, workspace):
        from molexp.agent.sessions.store import SessionStore
        from molexp.agent.sessions.types import SessionMetadata
        from molexp.agent.types import Goal, SessionStatus

        store = SessionStore(workspace.subsystem_store("agent.sessions").dir())
        meta = SessionMetadata(
            session_id="sess-x",
            goal=Goal(description="t"),
            status=SessionStatus.RUNNING,
        )
        store.write_metadata(meta)

        target = workspace.root / ".subsystems" / "agent.sessions" / "sess-x" / "session.json"
        assert target.exists()


class TestMcpStoreWiring:
    def test_mcp_store_writes_under_subsystem_dir(self, workspace, tmp_path):
        from molexp.agent.mcp.store import McpScope, McpStore

        user_home = tmp_path / "user_home"
        store = McpStore(
            workspace.subsystem_store("agent.mcp").dir(),
            user_home_dir=user_home,
        )
        store.upsert(
            McpScope.WORKSPACE,
            "demo",
            {"type": "stdio", "command": "echo", "args": ["hi"]},
        )

        target = workspace.root / ".subsystems" / "agent.mcp" / "mcp.json"
        assert target.exists()
        payload = json.loads(target.read_text())
        assert "mcpServers" in payload
        assert "demo" in payload["mcpServers"]
