"""Asset primitives for molexp runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .context import require_current_context


@dataclass(slots=True)
class Asset:
    """Minimal record of a produced resource."""

    uri: str
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_path(self) -> Path:
        """Interpret ``uri`` as a local filesystem path."""

        return Path(self.uri)


class AssetRepo:
    """In-memory registry of :class:`Asset` instances."""

    def __init__(self) -> None:
        self._by_uri: dict[str, Asset] = {}

    def add(self, asset: Asset) -> Asset:
        """Store ``asset`` keyed by its uri, replacing any existing entry."""

        self._by_uri[asset.uri] = asset
        return asset

    def list(self) -> list[Asset]:
        """Return a snapshot list of stored assets."""

        return list(self._by_uri.values())

    def get_by_uri(self, uri: str | Path) -> Asset | None:
        """Return the asset registered for ``uri`` if present."""

        return self._by_uri.get(str(uri))

    def clear(self) -> None:
        """Remove all tracked assets."""

        self._by_uri.clear()


def register_asset(
    uri: str | Path,
    *,
    label: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Asset:
    """Register an :class:`Asset` in the current run context."""

    ctx = require_current_context()
    asset = Asset(uri=str(uri), label=label, meta=dict(meta or {}))
    return ctx.asset_repo.add(asset)

