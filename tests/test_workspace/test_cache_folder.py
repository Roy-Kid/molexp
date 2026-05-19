"""Tests for ``CacheFolder`` and its ``CacheStore`` adapter.

Sub-spec ``unify-folder-abstraction-03-system-folder-migration`` adds
``CacheFolder`` rooted at ``<workspace_root>/cache/`` (no more dotfile
``.subsystems/workflow.cache/`` indirection). The folder exposes a
schema-agnostic file API (``read_entry`` / ``write_entry`` / ``keys``
/ ``total_bytes`` / ``clear``) plus an ``as_cache_store()`` method that
returns an adapter satisfying the workflow-layer
:class:`molexp.workflow.cache_store.CacheStore` Protocol — without
importing workflow at module load.

Spec acceptance criteria: **ac-002**, **ac-003**.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.workflow import Caching, TaskSnapshot
from molexp.workflow.cache_store import CacheStore
from molexp.workspace import Workspace
from molexp.workspace.cache import (
    WORKSPACE_CACHE_KIND,
    CacheFolder,
)


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


# ── entry_path / read / write contract ────────────────────────────────────────


def test_entry_path_is_under_cache_dir(folder: CacheFolder, workspace: Workspace) -> None:
    assert str(folder.entry_path("k")) == str(workspace.root / "cache" / "k.json")


def test_read_entry_miss_returns_none(folder: CacheFolder) -> None:
    assert folder.read_entry("missing") is None


def test_write_then_read_entry_round_trips(folder: CacheFolder) -> None:
    """A write/read pair preserves the JSON value (formatting may normalize)."""
    import json

    folder.write_entry("k", '{"hello": "world"}')
    raw = folder.read_entry("k")
    assert raw is not None
    assert json.loads(raw) == {"hello": "world"}


def test_write_entry_is_atomic(folder: CacheFolder, workspace: Workspace) -> None:
    folder.write_entry("k", '{"y": 1}')
    cache_dir = Path(str(workspace.root / "cache"))
    leftover = list(cache_dir.glob("*.tmp"))
    assert not leftover, f"expected no leftover tmp files; got {leftover}"


# ── keys / total_bytes / clear ────────────────────────────────────────────────


def test_keys_yields_stems_of_json_files(folder: CacheFolder) -> None:
    folder.write_entry("a", '{"x": 1}')
    folder.write_entry("b", '{"y": 2}')
    assert sorted(folder.keys()) == ["a", "b"]


def test_total_bytes_sums_entry_sizes(folder: CacheFolder, workspace: Workspace) -> None:
    folder.write_entry("a", '{"x": 1}')
    folder.write_entry("b", '{"y": 2}')
    expected = sum(
        p.stat().st_size for p in Path(str(workspace.root / "cache")).glob("*.json")
    )
    assert folder.total_bytes() == expected


def test_clear_removes_all_entries_and_returns_count(
    folder: CacheFolder, workspace: Workspace
) -> None:
    folder.write_entry("a", '{"x": 1}')
    folder.write_entry("b", '{"y": 2}')
    removed = folder.clear()
    assert removed == 2
    assert list(Path(str(workspace.root / "cache")).glob("*.json")) == []


def test_clear_on_empty_returns_zero(folder: CacheFolder) -> None:
    assert folder.clear() == 0


# ── kind constant ─────────────────────────────────────────────────────────────


def test_kind_constant_is_workspace_namespaced() -> None:
    assert WORKSPACE_CACHE_KIND == "workspace.cache"


# ── as_cache_store: CacheStore Protocol compliance ────────────────────────────


def test_as_cache_store_returns_cachestore_protocol_instance(folder: CacheFolder) -> None:
    """The adapter must structurally satisfy the workflow Protocol."""
    store = folder.as_cache_store()
    assert isinstance(store, CacheStore)


def test_caching_round_trip_via_workspace_cache(
    workspace: Workspace, snapshot: TaskSnapshot
) -> None:
    """A ``Caching`` instance backed by ``ws.Cache().as_cache_store()`` round-trips."""
    cache = Caching(store=workspace.cache.as_cache_store())
    cache.put(snapshot, inputs={"x": 1}, result={"y": 2})

    hit = cache.get(snapshot, inputs={"x": 1})
    assert hit == {"y": 2}

    # And the file lives under <root>/cache/<key>.json, never under .subsystems/.
    entries = list(Path(str(workspace.root / "cache")).glob("*.json"))
    assert len(entries) == 1, f"expected one entry under <root>/cache; got {entries}"
    assert not Path(str(workspace.root / ".subsystems" / "workflow.cache")).exists()


def test_cache_folder_lazy_mkdir(workspace: Workspace) -> None:
    """``Workspace.Cache()`` should not create ``<root>/cache/`` until first touch."""
    workspace.cache  # idempotent vend  # noqa: B018
    assert not Path(str(workspace.root / "cache")).exists()


# ── as_cache_store lazy-imports workflow ──────────────────────────────────────


def test_workspace_cache_folder_import_does_not_load_workflow() -> None:
    """``import molexp.workspace.cache.folder`` must not pull workflow.

    This is the layer-charter guard: the workspace cache folder must stay
    independent of the workflow package at module-load time. Only when
    a caller asks for ``as_cache_store()`` does the workflow Protocol
    become (typing-only) relevant.
    """
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
