"""Workspace asset catalog — moved here from ``workspace/assets/``.

``AssetCatalog`` (the class body) now lives in
:mod:`molexp.workspace.catalog.index`. ``CatalogFolder`` wraps the
on-disk directory ``<workspace_root>/catalog/`` (no dotfile prefix).
The legacy import ``from molexp.workspace.assets import AssetCatalog``
keeps working through a re-export shim in ``assets/catalog.py``.
"""

from molexp.workspace.catalog.folder import (
    WORKSPACE_CATALOG_KIND,
    CatalogFolder,
)
from molexp.workspace.catalog.index import (
    CATALOG_DIRNAME,
    CATALOG_FILENAME,
    CATALOG_SCHEMA_VERSION,
    AssetCatalog,
    RebuildReport,
)

__all__ = [
    "CATALOG_DIRNAME",
    "CATALOG_FILENAME",
    "CATALOG_SCHEMA_VERSION",
    "WORKSPACE_CATALOG_KIND",
    "AssetCatalog",
    "CatalogFolder",
    "RebuildReport",
]
