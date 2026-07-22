"""Tests for ``CacheFolder`` and its ``CacheStore`` adapter.

``CacheFolder`` is rooted at ``<workspace_root>/cache/`` and exposes a
schema-agnostic file API (``read_entry`` / ``write_entry`` / ``keys`` /
``total_bytes`` / ``clear``) plus an ``as_cache_store()`` adapter satisfying the
workflow-layer :class:`molexp.workflow.cache_store.CacheStore` Protocol —
without importing workflow at module load (the layer charter).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.workflow import Caching, TaskSnapshot
from molexp.workspace import Workspace
from molexp.workspace.cache import CacheFolder


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab")
    ws.materialize()
    return ws


@pytest.fixture
def folder(workspace: Workspace) -> CacheFolder:
    return workspace.cache


@pytest.fixture
def snapshot() -> TaskSnapshot:
    return TaskSnapshot(
        key="task:fixture:hash",
        task_id="t1",
        task_type="ExampleTask",
        code_hash="codehash",
        config_hash="confighash",
        created_at=datetime.now(UTC),
    )


class TestCacheFolder:
    """The schema-agnostic file API + the ``CacheStore`` adapter."""

    def test_read_entry_miss_returns_none(self, folder: CacheFolder) -> None:
        assert folder.read_entry("missing") is None

    def test_write_then_read_entry_round_trips(self, folder: CacheFolder) -> None:
        folder.write_entry("k", '{"hello": "world"}')
        raw = folder.read_entry("k")
        assert raw is not None
        assert json.loads(raw) == {"hello": "world"}

    def test_keys_yields_stems_of_json_files(self, folder: CacheFolder) -> None:
        folder.write_entry("a", '{"x": 1}')
        folder.write_entry("b", '{"y": 2}')
        assert sorted(folder.keys()) == ["a", "b"]

    def test_clear_removes_all_entries_and_returns_count(
        self, folder: CacheFolder, workspace: Workspace
    ) -> None:
        folder.write_entry("a", '{"x": 1}')
        folder.write_entry("b", '{"y": 2}')
        removed = folder.clear()
        assert removed == 2
        assert list(Path(str(workspace.root / "cache")).glob("*.json")) == []

    def test_lazy_mkdir_defers_dir_until_first_touch(self, workspace: Workspace) -> None:
        workspace.cache  # idempotent vend  # noqa: B018
        assert not Path(str(workspace.root / "cache")).exists()

    def test_as_cache_store_round_trips_through_caching(
        self, workspace: Workspace, snapshot: TaskSnapshot
    ) -> None:
        """A ``Caching`` backed by ``ws.cache.as_cache_store()`` round-trips, with
        entries under ``<root>/cache/`` (never a legacy ``.subsystems/`` dir)."""
        cache = Caching(store=workspace.cache.as_cache_store())
        cache.put(snapshot, inputs={"x": 1}, result={"y": 2})

        hit = cache.get(snapshot, inputs={"x": 1})
        assert hit == {"y": 2}

        entries = list(Path(str(workspace.root / "cache")).glob("*.json"))
        assert len(entries) == 1, f"expected one entry under <root>/cache; got {entries}"
        assert not Path(str(workspace.root / ".subsystems" / "workflow.cache")).exists()


def test_cache_folder_import_does_not_load_workflow() -> None:
    """Layer-charter guard: ``import molexp.workspace.cache.folder`` must not pull
    ``molexp.workflow`` into ``sys.modules`` at module-load time. Only a caller
    asking for ``as_cache_store()`` makes the workflow Protocol (typing-only)
    relevant."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import molexp.workspace.cache.folder  # noqa: F401\n"
        "assert 'molexp.workflow' not in sys.modules, "
        "    'molexp.workspace.cache.folder eagerly imported molexp.workflow'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
