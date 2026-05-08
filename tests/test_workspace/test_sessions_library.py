"""Tests for ``Workspace.sessions`` (spec ac-008).

The session library is workspace's mediator for the agent layer's
on-disk session metadata + the catalog's ``sessions`` section. Each
write goes to both: ``session.json`` under
``<root>/.subsystems/agent.sessions/<id>/`` and a catalog row in the
``sessions`` section.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path)
    ws.materialize()
    return ws


@pytest.fixture
def session_metadata() -> dict:
    """Plain-dict SessionMetadata projection — duck-typed input."""
    return {
        "session_id": "sess-123",
        "status": "active",
        "created_at": "2026-05-08T10:00:00Z",
        "updated_at": "2026-05-08T10:00:00Z",
        "summary": "smoke test",
        "goal": {"description": "investigate the convergence"},
    }


class TestSessionsLibraryAccess:
    def test_workspace_exposes_sessions_property(self, workspace):
        sessions = workspace.sessions
        assert sessions is not None

    def test_sessions_is_cached(self, workspace):
        assert workspace.sessions is workspace.sessions


class TestSessionsLibraryCreate:
    def test_create_writes_session_json_to_subsystem_dir(self, workspace, session_metadata):
        workspace.sessions.create(session_metadata)
        path = workspace.root / ".subsystems" / "agent.sessions" / "sess-123" / "session.json"
        assert path.exists()
        on_disk = json.loads(path.read_text())
        assert on_disk["session_id"] == "sess-123"

    def test_create_adds_catalog_row(self, workspace, session_metadata):
        workspace.sessions.create(session_metadata)
        rows = workspace.catalog.query_sessions()
        assert len(rows) == 1
        assert rows[0]["session_id"] == "sess-123"
        assert rows[0]["workspace_id"] == workspace.id
        assert rows[0]["status"] == "active"
        assert rows[0]["goal_summary"] == "investigate the convergence"

    def test_create_accepts_pydantic_metadata(self, workspace):
        # Duck-typed: any object with ``model_dump`` works.
        from pydantic import BaseModel

        class FakeMeta(BaseModel):
            session_id: str
            status: str
            summary: str = ""

        workspace.sessions.create(FakeMeta(session_id="pyd-1", status="active"))
        rows = workspace.catalog.query_sessions()
        assert {r["session_id"] for r in rows} == {"pyd-1"}


class TestSessionsLibraryRead:
    def test_list_returns_catalog_rows(self, workspace, session_metadata):
        workspace.sessions.create(session_metadata)
        listed = workspace.sessions.list()
        assert {row["session_id"] for row in listed} == {"sess-123"}

    def test_get_returns_row_by_id(self, workspace, session_metadata):
        workspace.sessions.create(session_metadata)
        row = workspace.sessions.get("sess-123")
        assert row is not None
        assert row["session_id"] == "sess-123"

    def test_get_returns_none_for_unknown_id(self, workspace):
        assert workspace.sessions.get("does-not-exist") is None


class TestSessionsLibraryDelete:
    def test_delete_removes_catalog_row(self, workspace, session_metadata):
        workspace.sessions.create(session_metadata)
        workspace.sessions.delete("sess-123")
        assert workspace.sessions.list() == []

    def test_delete_removes_session_directory(self, workspace, session_metadata):
        workspace.sessions.create(session_metadata)
        session_dir = workspace.root / ".subsystems" / "agent.sessions" / "sess-123"
        assert session_dir.exists()
        workspace.sessions.delete("sess-123")
        assert not session_dir.exists()

    def test_delete_unknown_session_is_silent(self, workspace):
        workspace.sessions.delete("never-existed")  # must not raise


class TestSessionsLibraryNoAgentImport:
    """ac-008 / ac-011: workspace/sessions.py imports zero molexp.agent symbols."""

    def test_sessions_module_has_no_molexp_agent_imports(self):
        path = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workspace" / "sessions.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molexp.agent"), alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("molexp.agent"), node.module
