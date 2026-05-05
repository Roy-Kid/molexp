"""Typed asset model.

This package defines the Asset class hierarchy, per-scope manifest,
and workspace-level catalog.  See
``docs/development/specs/unified-asset-model.md`` for the design.
"""

from . import lineage
from ._adapter import ASSET_ADAPTER, AnyAsset, parse_asset
from .accessors import ArtifactAccessor, CheckpointAccessor, LogAccessor
from .artifact import ArtifactAsset
from .base import Asset, AssetScope, Producer
from .catalog import AssetCatalog
from .checkpoint import CheckpointAsset
from .data import DataAsset, DataAssetLibrary, ImportAction
from .error import ErrorTraceAsset
from .execution import ExecutionStateAsset
from .log import LogAsset
from .manifest import AssetManifest
from .output import OutputAsset
from .view import AssetsView

__all__ = [
    "ASSET_ADAPTER",
    "AnyAsset",
    "ArtifactAccessor",
    "ArtifactAsset",
    "Asset",
    "AssetCatalog",
    "AssetManifest",
    "AssetScope",
    "AssetsView",
    "CheckpointAccessor",
    "CheckpointAsset",
    "DataAsset",
    "DataAssetLibrary",
    "ErrorTraceAsset",
    "ImportAction",
    "ExecutionStateAsset",
    "LogAccessor",
    "LogAsset",
    "OutputAsset",
    "Producer",
    "lineage",
    "parse_asset",
]
