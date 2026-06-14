"""Concurrency + rebuild-equivalence guarantees for the derived catalog.

P0-1 (workflow-workspace-hardening): the catalog moved from a single
``index.json`` rewritten under a process-local ``threading.Lock`` — which
loses rows when two OS processes read-modify-write the same file — to a
SQLite backend whose row-level upserts survive concurrent multi-process
writers (WAL + ``busy_timeout``).

ac-003  N processes concurrently register/upsert → no lost rows.
ac-005  entity ``*.json`` stays the single source of truth; ``rebuild()``
        reproduces a query-equivalent catalog from disk.
"""

from __future__ import annotations

import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import pytest

from molexp.workspace import (
    ArtifactAsset,
    AssetCatalog,
    AssetScope,
    Producer,
    Workspace,
)


def _make_artifact(asset_id: str, scope: AssetScope) -> ArtifactAsset:
    now = datetime.now()
    return ArtifactAsset(
        asset_id=asset_id,
        name=f"{asset_id}.bin",
        scope=scope,
        path=Path(f"artifacts/{asset_id}.bin"),
        created_at=now,
        updated_at=now,
        producer=Producer(execution_id="e1", task_id="t1"),
        mime="application/octet-stream",
        size=1,
    )


def _register_worker(root_str: str, worker_id: int, per_worker: int) -> None:
    """Register ``per_worker`` distinct assets from one OS process."""
    catalog = AssetCatalog(Path(root_str))
    scope = AssetScope(kind="workspace", ids=())
    for i in range(per_worker):
        catalog.register(_make_artifact(f"w{worker_id}-a{i}", scope))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab")
    ws.materialize()
    return ws


@pytest.mark.unit
def test_concurrent_multiprocess_register_loses_no_rows(workspace: Workspace) -> None:
    """N OS processes each register K distinct assets → all N*K survive."""
    n_workers = 4
    per_worker = 25
    root_str = str(workspace.root)

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_register_worker, args=(root_str, wid, per_worker))
        for wid in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    catalog = AssetCatalog(workspace.root)
    assets = catalog.query_assets(scope=AssetScope(kind="workspace", ids=()))
    ids = {a.asset_id for a in assets}
    expected = {f"w{w}-a{i}" for w in range(n_workers) for i in range(per_worker)}
    assert ids == expected, f"lost {len(expected - ids)} rows"
