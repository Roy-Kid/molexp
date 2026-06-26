"""Workspace curation — scan, query, reorganize, and dedupe a workspace tree.

Pure functions that **compose** existing workspace primitives (``AssetCatalog``,
``AssetsView``, ``DataAssetLibrary``, ``Folder.move_to`` / ``Folder.delete``) —
they add no new scanning, hashing, move, or import machinery. The harness
surfaces these as toolified capabilities (see the curation capability catalog);
this layer is the storage-level building block they bind to.

Kept out of the top-level ``molexp.workspace.__all__`` so ``import
molexp.workspace`` stays light; consumers import ``molexp.workspace.curation``
directly.
"""

from __future__ import annotations

from molexp.workspace.curation.consolidate import (
    consolidate_workflow_source,
    dedupe_workflow_source,
)
from molexp.workspace.curation.inventory import (
    ExperimentInventory,
    ProjectInventory,
    RunInventory,
    WorkspaceInventory,
    scan_workspace,
)
from molexp.workspace.curation.query import aggregate_assets_by_kind, find_asset_by_hash
from molexp.workspace.curation.reorg import delete_folder, move_run, rehome_asset

__all__ = [
    "ExperimentInventory",
    "ProjectInventory",
    "RunInventory",
    "WorkspaceInventory",
    "aggregate_assets_by_kind",
    "consolidate_workflow_source",
    "dedupe_workflow_source",
    "delete_folder",
    "find_asset_by_hash",
    "move_run",
    "rehome_asset",
    "scan_workspace",
]
