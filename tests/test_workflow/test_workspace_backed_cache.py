"""``Caching`` over ``ws.cache.as_cache_store()`` writes under ``<root>/cache/``.

Verifies the unify-folder-abstraction sub-spec 03 contract — workflow's
cache sits inside the workspace it serves (via the singleton ``CacheFolder``
exposed at ``ws.cache``), not in ``~/.molexp/cache/``. Also pins the
``Caching`` constructor's store/store_dir XOR validation. Touches the public
surface only (``Caching``, the ``CacheStore`` adapter returned by
``ws.cache.as_cache_store()``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.workflow import (
    Caching,
    TaskSnapshot,
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


class TestCaching:
    def test_workspace_backed_store_writes_under_cache_dir(
        self, workspace: Workspace, snapshot: TaskSnapshot
    ) -> None:
        cache = Caching(store=workspace.cache.as_cache_store())
        cache.put(snapshot, inputs={"x": 1}, result={"y": 2})

        expected_dir = Path(workspace.root / "cache")
        assert expected_dir.exists(), (
            f"ws.cache.as_cache_store() should write under {expected_dir}, but the dir is missing"
        )
        files = list(expected_dir.glob("*.json"))
        assert len(files) == 1, f"expected exactly one cache entry; got {len(files)}"

    def test_constructor_rejects_neither_store_nor_dir(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Caching()  # type: ignore[call-arg]

    def test_constructor_rejects_both_store_and_dir(
        self, workspace: Workspace, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="not accept both"):
            Caching(store=workspace.cache.as_cache_store(), store_dir=tmp_path / "fs-cache")
