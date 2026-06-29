"""Typed asset model.

This package defines the Asset class hierarchy, the per-scope ``assets.json``
manifest, and the manifest-scanning query layer (:mod:`.scan`).  See
``docs/development/specs/unified-asset-model.md`` for the design.
"""

from . import lineage, scan
from ._adapter import ASSET_ADAPTER, AnyAsset, parse_asset
from .accessors import ArtifactAccessor, CheckpointAccessor, LogAccessor
from .artifact import ArtifactAsset
from .base import Asset, AssetScope, Producer
from .checkpoint import CheckpointAsset
from .data import DataAsset, DataAssetLibrary, ImportAction
from .error import ErrorTraceAsset
from .log import LogAsset
from .manifest import AssetManifest
from .view import AssetsView

__all__ = [
    "ASSET_ADAPTER",
    "AnyAsset",
    "ArtifactAccessor",
    "ArtifactAsset",
    "Asset",
    "AssetManifest",
    "AssetScope",
    "AssetsView",
    "CheckpointAccessor",
    "CheckpointAsset",
    "DataAsset",
    "DataAssetLibrary",
    "ErrorTraceAsset",
    "ImportAction",
    "LogAccessor",
    "LogAsset",
    "Producer",
    "lineage",
    "parse_asset",
    "scan",
]
