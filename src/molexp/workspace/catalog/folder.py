"""``CatalogFolder`` — typed wrapper around the on-disk asset catalog dir.

Rooted at ``<workspace_root>/catalog/`` (no dotfile prefix). The
underlying single-file index lives at ``<root>/catalog/index.json`` and
is owned by :class:`molexp.workspace.catalog.index.AssetCatalog`.
"""

from __future__ import annotations

from pathlib import Path

from ..folder import Folder
from .index import CATALOG_FILENAME, AssetCatalog

WORKSPACE_CATALOG_KIND = "workspace.catalog"


class CatalogFolder(Folder):
    """Single-instance folder hosting the workspace asset catalog.

    Construction is side-effect-free. The on-disk directory is materialized
    lazily on first :class:`AssetCatalog` write through :attr:`catalog`.
    """

    @property
    def catalog(self) -> AssetCatalog:
        """Return an :class:`AssetCatalog` rooted at this folder's parent.

        The catalog vendors its own atomic writes under
        ``<root>/catalog/index.json``; this property is a typed
        convenience so callers do not have to thread the workspace root
        through manually.
        """
        # AssetCatalog accepts the workspace root and derives <root>/catalog
        # itself, matching the legacy CATALOG_DIRNAME contract.
        return AssetCatalog(self._workspace_root)

    @property
    def index_path(self) -> Path:
        return self._workspace_root / "catalog" / CATALOG_FILENAME

    @property
    def _workspace_root(self) -> Path:
        # ``CatalogFolder`` is constructed as a direct child of ``Workspace``,
        # so the parent's path is the workspace root.
        parent = self.parent
        if parent is None:
            raise RuntimeError("CatalogFolder must be a child of Workspace")
        return parent._compute_path()


__all__ = [
    "WORKSPACE_CATALOG_KIND",
    "CatalogFolder",
]
