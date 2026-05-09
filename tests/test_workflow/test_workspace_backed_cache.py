"""``Caching`` over a ``WorkspaceCacheStore`` writes under workspace's subsystem dir.

Verifies the rectification spec's Phase 2 contract — workflow's cache
sits inside the workspace it serves, not in ``~/.molexp/cache/``.
Touches the public surface only (``Caching``, ``WorkspaceCacheStore``,
``WORKFLOW_CACHE_SUBSYSTEM_KIND``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.workflow import (
    WORKFLOW_CACHE_SUBSYSTEM_KIND,
    Caching,
    TaskSnapshot,
    WorkspaceCacheStore,
)
from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab")
    ws.materialize()
    return ws


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


def test_workspace_backed_cache_writes_under_subsystem_dir(
    workspace: Workspace, snapshot: TaskSnapshot
) -> None:
    cache = Caching(store=WorkspaceCacheStore(workspace))
    cache.put(snapshot, inputs={"x": 1}, result={"y": 2})

    expected_dir = workspace.root / ".subsystems" / WORKFLOW_CACHE_SUBSYSTEM_KIND
    assert expected_dir.exists(), (
        f"WorkspaceCacheStore should write under {expected_dir}, but the dir is missing"
    )
    files = list(expected_dir.glob("*.json"))
    assert len(files) == 1, f"expected exactly one cache entry; got {len(files)}"


def test_workspace_backed_cache_round_trip(workspace: Workspace, snapshot: TaskSnapshot) -> None:
    cache = Caching(store=WorkspaceCacheStore(workspace))
    cache.put(snapshot, inputs={"x": 1}, result={"y": 2})

    hit = cache.get(snapshot, inputs={"x": 1})
    assert hit == {"y": 2}


def test_workspace_backed_cache_different_inputs_miss(
    workspace: Workspace, snapshot: TaskSnapshot
) -> None:
    cache = Caching(store=WorkspaceCacheStore(workspace))
    cache.put(snapshot, inputs={"x": 1}, result={"y": 2})

    assert cache.get(snapshot, inputs={"x": 999}) is None


def test_workspace_cache_entry_is_valid_json(workspace: Workspace, snapshot: TaskSnapshot) -> None:
    cache = Caching(store=WorkspaceCacheStore(workspace))
    cache.put(snapshot, inputs={"x": 1}, result={"y": 2})

    entry_path = next(
        (workspace.root / ".subsystems" / WORKFLOW_CACHE_SUBSYSTEM_KIND).glob("*.json")
    )
    payload = json.loads(entry_path.read_text())
    assert payload["snapshot_key"] == snapshot.key
    assert payload["task_id"] == "t1"
    assert payload["result"] == {"y": 2}


def test_caching_constructor_rejects_neither_store_nor_dir() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Caching()  # type: ignore[call-arg]


def test_caching_constructor_rejects_both_store_and_dir(
    workspace: Workspace, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="not accept both"):
        Caching(store=WorkspaceCacheStore(workspace), store_dir=tmp_path / "fs-cache")


def test_workspace_subsystem_kind_constant_is_namespaced() -> None:
    # The kind name is an *agent-level* convention: workspace just
    # vends it. The constant is exported as a stability anchor.
    assert WORKFLOW_CACHE_SUBSYSTEM_KIND == "workflow.cache"
