"""Headline benchmark for workflow-workspace-hardening P0-1 (ac-004).

Measures per-asset ``AssetCatalog.register`` cost as the catalog already
holds ``A`` assets, for growing ``A``.

The legacy single-``index.json`` backend rewrote the whole file on every
register, so a single insert cost ~O(A) and filling the catalog was
quadratic. The SQLite backend does a row-level ``INSERT OR REPLACE``, so a
single insert is ~O(log A) and per-register time stays roughly flat as the
store grows. This script reports the per-register microseconds at each fill
level so the before/after curve is directly comparable. It is a MEASUREMENT
harness, not a test: it never asserts, it only prints.

Run::

    python -m benches.bench_catalog_register
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path

from molexp.workspace.assets import ArtifactAsset, AssetScope, Producer
from molexp.workspace.catalog.index import AssetCatalog

# Catalog fill levels (number of assets already present) to probe.
FILL_LEVELS = (0, 250, 500, 1000, 2000, 4000)
# Assets registered (and timed) at each fill level.
SAMPLE = 200


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


def _fill_to(catalog: AssetCatalog, scope: AssetScope, current: int, target: int) -> None:
    for i in range(current, target):
        catalog.register(_make_artifact(f"fill-{i}", scope))


def _time_sample(catalog: AssetCatalog, scope: AssetScope, base: int, n: int) -> float:
    """Return mean microseconds per register over ``n`` fresh assets."""
    start = time.perf_counter()
    for i in range(n):
        catalog.register(_make_artifact(f"probe-{base}-{i}", scope))
    elapsed = time.perf_counter() - start
    return (elapsed / n) * 1e6


def main() -> None:
    scope = AssetScope(kind="workspace", ids=())
    with tempfile.TemporaryDirectory() as tmp:
        catalog = AssetCatalog(Path(tmp) / "lab")
        print(f"{'fill A':>10} {'per-register µs':>18}")
        print("-" * 30)
        present = 0
        for level in FILL_LEVELS:
            _fill_to(catalog, scope, present, level)
            present = level
            per_register = _time_sample(catalog, scope, level, SAMPLE)
            present += SAMPLE
            print(f"{level:>10} {per_register:>18.1f}")
        print(
            "\nSQLite row upsert keeps per-register time ~flat as A grows;\n"
            "the legacy whole-file rewrite grew linearly with A."
        )


if __name__ == "__main__":
    main()
