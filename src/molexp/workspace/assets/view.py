"""Scope-bound catalog view — ``{scope}.assets``.

Returned by ``Workspace.assets`` / ``Project.assets`` /
``Experiment.assets`` / ``Run.assets``.  Presents the scope as a
read-only filtered view of the workspace ``AssetCatalog``.

For importing ``DataAsset`` inputs, use ``{scope}.data_assets`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from .base import Asset, AssetScope

if TYPE_CHECKING:
    from .catalog import AssetCatalog


# Alias avoids the static-checker confusion where the ``list`` method name
# shadows the ``list`` builtin in return-type expressions.
AssetList: TypeAlias = list[Asset]


class AssetsView:
    """Read-only, scope-filtered view of the workspace catalog."""

    def __init__(self, catalog: "AssetCatalog", scope: AssetScope) -> None:
        self._catalog = catalog
        self._scope = scope

    def list(self) -> AssetList:
        return self._catalog.query_assets(scope=self._scope)

    # Alias that mirrors the old API surface.
    def list_assets(self) -> AssetList:
        return self.list()

    def get(self, asset_id: str) -> Asset | None:
        asset = self._catalog.get(asset_id)
        if asset is None:
            return None
        if asset.scope != self._scope:
            return None
        return asset

    def query(
        self,
        *,
        kind: str | type[Asset] | None = None,
        producer_run: str | None = None,
        producer_task: str | None = None,
        tag: tuple[str, str] | None = None,
        limit: int | None = None,
        recursive: bool = False,
    ) -> AssetList:
        """Filtered asset query against the catalog at this scope.

        When ``recursive`` is ``True``, matches assets in any sub-scope
        underneath this view's scope — for instance an
        ``experiment.assets.query(recursive=True)`` returns assets
        produced by every run in the experiment.
        """
        return self._catalog.query_assets(
            kind=kind,
            scope=self._scope,
            producer_run=producer_run,
            producer_task=producer_task,
            tag=tag,
            limit=limit,
            recursive=recursive,
        )
