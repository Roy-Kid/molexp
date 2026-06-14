"""Workspace asset catalog — moved here from ``workspace/assets/``.

``AssetCatalog`` (the class body) lives in
:mod:`molexp.workspace.catalog.index` and owns the on-disk directory
``<workspace_root>/catalog/`` (no dotfile prefix). ``ws.catalog`` returns
an :class:`AssetCatalog` directly. The legacy import
``from molexp.workspace.assets import AssetCatalog`` keeps working through a
re-export shim in ``assets/catalog.py``.
"""

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
    "AssetCatalog",
    "RebuildReport",
]
