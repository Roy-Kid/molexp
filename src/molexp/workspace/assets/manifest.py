"""Per-scope asset manifest — ``assets.json`` in each scope directory.

Layout::

    {
      "schema_version": 1,
      "assets": {
        "<asset_id>": { ...Asset serialization... },
        ...
      }
    }

Writes go through a process-local lock + atomic rename so concurrent
tasks inside a single run process can append assets safely.  Cross-process
coordination is out of scope (see spec §2).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

from ._adapter import ASSET_ADAPTER, parse_asset
from .base import Asset

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "assets.json"

# Alias avoids the static-checker confusion where ``list`` (the method) shadows
# ``list`` (the builtin) in return annotations.
type AssetList = list[Asset]


class AssetManifest:
    """JSON-backed dict of assets for one scope.

    ``scope_dir`` is coerced to :class:`pathlib.Path` because the manifest
    does genuine local I/O (atomic rename, file locking) — callers may
    pass :class:`molexp.Path` or :class:`str`.
    """

    def __init__(self, scope_dir: str | PathLike[str]) -> None:
        self.scope_dir = Path(scope_dir)
        self.path = self.scope_dir / MANIFEST_FILENAME
        self._lock = threading.Lock()

    # ── Read ──────────────────────────────────────────────────────────────

    def load(self) -> dict[str, Asset]:
        """Return a fresh mapping ``{asset_id -> Asset}`` from disk."""
        if not self.path.exists():
            return {}
        with open(self.path) as fh:  # noqa: PTH123
            data = json.load(fh)
        raw_assets: dict = data.get("assets", {})
        return {aid: parse_asset(entry) for aid, entry in raw_assets.items()}

    def get(self, asset_id: str) -> Asset | None:
        return self.load().get(asset_id)

    def list(self) -> AssetList:
        return list(self.load().values())

    def __iter__(self) -> Iterator[Asset]:
        return iter(self.list())

    # ── Write ─────────────────────────────────────────────────────────────

    def register(self, asset: Asset) -> None:
        """Insert ``asset`` into the manifest (overwrites if asset_id exists)."""
        with self._lock:
            assets = self._load_raw()
            assets[asset.asset_id] = _dump(asset)
            self._save_raw(assets)

    def update(self, asset: Asset) -> None:
        """Replace an existing entry.  Raises ``KeyError`` if missing."""
        with self._lock:
            assets = self._load_raw()
            if asset.asset_id not in assets:
                raise KeyError(f"Asset {asset.asset_id!r} not in manifest at {self.path}")
            assets[asset.asset_id] = _dump(asset)
            self._save_raw(assets)

    def deregister(self, asset_id: str) -> None:
        with self._lock:
            assets = self._load_raw()
            assets.pop(asset_id, None)
            self._save_raw(assets)

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_raw(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path) as fh:  # noqa: PTH123
            data = json.load(fh)
        return dict(data.get("assets", {}))

    def _save_raw(self, assets: dict) -> None:
        from ..base import _atomic_write_json

        payload = {"schema_version": SCHEMA_VERSION, "assets": assets}
        _atomic_write_json(self.path, payload)


def _dump(asset: Asset) -> dict:
    return ASSET_ADAPTER.dump_python(asset, mode="json")
