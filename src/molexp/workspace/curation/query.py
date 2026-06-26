"""Asset queries over a workspace.

``find_asset_by_hash`` and ``aggregate_assets_by_kind`` are read-only
compositions over the ``AssetCatalog`` and per-scope ``AssetsView``. Neither
rebuilds the catalog: a directly-registered asset (e.g. an imported
``DataAsset``) lives only in the catalog, not in a scope manifest, so a rebuild
would drop it.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from molexp.workspace.assets.base import Asset
    from molexp.workspace.assets.view import AssetsView
    from molexp.workspace.workspace import Workspace

__all__ = ["aggregate_assets_by_kind", "find_asset_by_hash"]


class _AssetScope(Protocol):
    """A workspace entity exposing a per-scope ``AssetsView`` (Workspace /
    Project / Experiment / Run)."""

    @property
    def assets(self) -> AssetsView: ...


def find_asset_by_hash(workspace: Workspace, content_hash: str) -> Asset | None:
    """Return the asset with *content_hash*, or ``None`` if there is none.

    Composes ``workspace.catalog.find_by_content_hash``. Read-only.

    Args:
        workspace: The workspace whose catalog to search.
        content_hash: A content hash string (e.g. ``"sha256:..."``).

    Returns:
        The matching :class:`Asset`, or ``None``.
    """
    return workspace.catalog.find_by_content_hash(content_hash)


def aggregate_assets_by_kind(scope: _AssetScope, *, recursive: bool = False) -> dict[str, int]:
    """Count the assets in *scope* keyed by their ``kind``.

    Composes ``scope.assets.query`` (no rebuild). Read-only.

    Args:
        scope: Any workspace entity exposing ``.assets`` (Workspace / Project /
            Experiment / Run).
        recursive: When ``True``, include assets registered in sub-scopes.

    Returns:
        A ``{kind: count}`` mapping; empty when the scope holds no assets.
    """
    # ``kind`` is the discriminated-union tag on every concrete asset subclass;
    # the base ``Asset`` query() returns doesn't declare it (repo-wide pattern).
    counts: Counter[str] = Counter(
        asset.kind  # ty: ignore[unresolved-attribute]
        for asset in scope.assets.query(recursive=recursive)
    )
    return dict(counts)
