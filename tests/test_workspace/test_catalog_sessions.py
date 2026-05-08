"""Tests for ``AssetCatalog`` sessions section (spec ac-003 / ac-004 / ac-005).

Sessions are workspace-flat first-class entities sitting next to
``runs`` / ``executions`` in the catalog. Schema bumps to version 2;
``run_id`` is a soft FK that does **not** cascade.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from molexp.workspace.assets import AssetCatalog
from molexp.workspace.assets.catalog import (
    _EMPTY_CATALOG,
    CATALOG_SCHEMA_VERSION,
)


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    (tmp_path / "workspace.json").write_text('{"id": "ws", "name": "ws"}')
    return tmp_path


@pytest.fixture
def session_record() -> dict:
    return {
        "session_id": "sess-1",
        "workspace_id": "ws",
        "status": "active",
        "goal_summary": "Investigate convergence",
        "created_at": "2026-05-08T10:00:00Z",
        "updated_at": "2026-05-08T10:00:00Z",
        "run_id": None,
    }


class TestSchemaBump:
    def test_catalog_schema_version_is_two(self):
        assert CATALOG_SCHEMA_VERSION == 2

    def test_empty_catalog_includes_sessions_section(self):
        assert "sessions" in _EMPTY_CATALOG
        assert _EMPTY_CATALOG["sessions"] == {}


class TestUpsertSession:
    def test_upsert_persists_a_session_row(self, workspace_root, session_record):
        catalog = AssetCatalog(workspace_root)
        catalog.upsert_session(session_record)
        rows = catalog.query_sessions(workspace_id="ws")
        assert len(rows) == 1
        assert rows[0]["session_id"] == "sess-1"
        assert rows[0]["status"] == "active"
        assert rows[0]["goal_summary"] == "Investigate convergence"

    def test_upsert_is_idempotent_on_same_id(self, workspace_root, session_record):
        catalog = AssetCatalog(workspace_root)
        catalog.upsert_session(session_record)
        updated = {**session_record, "status": "complete", "summary_changed": True}
        catalog.upsert_session(updated)
        rows = catalog.query_sessions()
        assert len(rows) == 1
        assert rows[0]["status"] == "complete"


class TestQuerySessions:
    def test_query_sessions_filters_by_workspace_id(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        catalog.upsert_session(_make_session("sess-a", workspace_id="ws-a"))
        catalog.upsert_session(_make_session("sess-b", workspace_id="ws-b"))
        rows = catalog.query_sessions(workspace_id="ws-a")
        assert {r["session_id"] for r in rows} == {"sess-a"}

    def test_query_sessions_filters_by_status(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        catalog.upsert_session(_make_session("sess-a", status="active"))
        catalog.upsert_session(_make_session("sess-b", status="complete"))
        rows = catalog.query_sessions(status="complete")
        assert {r["session_id"] for r in rows} == {"sess-b"}

    def test_query_sessions_filters_by_run_id(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        catalog.upsert_session(_make_session("sess-a", run_id=None))
        catalog.upsert_session(_make_session("sess-b", run_id="run-1"))
        rows = catalog.query_sessions(run_id="run-1")
        assert {r["session_id"] for r in rows} == {"sess-b"}

    def test_query_sessions_supports_limit(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        for i in range(5):
            catalog.upsert_session(_make_session(f"sess-{i}"))
        rows = catalog.query_sessions(limit=2)
        assert len(rows) == 2


class TestRemoveSession:
    def test_remove_drops_only_that_session(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        catalog.upsert_session(_make_session("sess-a"))
        catalog.upsert_session(_make_session("sess-b"))
        catalog.remove_session("sess-a")
        rows = catalog.query_sessions()
        assert {r["session_id"] for r in rows} == {"sess-b"}

    def test_remove_does_not_cascade_to_runs(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        catalog.upsert_run(
            {
                "run_id": "run-1",
                "experiment_id": "exp-1",
                "path": "projects/p/experiments/e/runs/run-1",
                "status": "completed",
            }
        )
        catalog.upsert_session(_make_session("sess-a", run_id="run-1"))
        catalog.remove_session("sess-a")
        runs = catalog.query_runs()
        assert {r["run_id"] for r in runs} == {"run-1"}

    def test_remove_missing_session_is_silent(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        catalog.remove_session("nonexistent")  # must not raise


class TestRebuildSessionsFromDisk:
    def test_rebuild_scans_subsystem_dir_for_session_json(self, workspace_root):
        sessions_dir = workspace_root / ".subsystems" / "agent.sessions"
        for sid in ("alpha", "beta", "gamma"):
            session_dir = sessions_dir / sid
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": sid,
                        "status": "active",
                        "created_at": "2026-05-08T10:00:00Z",
                        "updated_at": "2026-05-08T10:00:00Z",
                        "summary": f"summary {sid}",
                        "goal": {"description": f"goal {sid}"},
                    }
                )
            )

        catalog = AssetCatalog(workspace_root)
        report = catalog.rebuild()

        assert report.sessions == 3
        rows = catalog.query_sessions()
        ids = {r["session_id"] for r in rows}
        assert ids == {"alpha", "beta", "gamma"}

    def test_rebuild_handles_missing_subsystem_dir(self, workspace_root):
        catalog = AssetCatalog(workspace_root)
        report = catalog.rebuild()  # must not raise
        assert report.sessions == 0
        assert catalog.query_sessions() == []


class TestImportBoundary:
    """ac-005: catalog.py must not import any molexp.agent symbol."""

    def test_catalog_module_has_no_molexp_agent_imports(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "molexp"
            / "workspace"
            / "assets"
            / "catalog.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molexp.agent"), alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("molexp.agent"), node.module


def _make_session(
    session_id: str,
    *,
    workspace_id: str = "ws",
    status: str = "active",
    run_id: str | None = None,
) -> dict:
    return {
        "session_id": session_id,
        "workspace_id": workspace_id,
        "status": status,
        "goal_summary": "",
        "created_at": "2026-05-08T10:00:00Z",
        "updated_at": "2026-05-08T10:00:00Z",
        "run_id": run_id,
    }
