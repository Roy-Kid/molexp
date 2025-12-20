"""Asset primitives for molexp runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from molexp.workflow.context import require_current_context


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
    """Register an :class:`Asset` in the current run context.

    If a workspace and run_metadata are available in the context, this will:
    1. Store the asset in the global asset repository (with deduplication)
    2. Create an AssetRef linking the run to the asset
    3. Update the run's asset_refs.json

    Otherwise, it behaves as before (in-memory only).
    """
    ctx = require_current_context()
    asset = Asset(uri=str(uri), label=label, meta=dict(meta or {}))
    ctx.asset_repo.add(asset)

    # If we have workspace integration, store in global repo
    if ctx.workspace and ctx.run_metadata:
        from datetime import datetime
        from pathlib import Path as PathLib

        from molexp.models import Asset as AssetModel
        from molexp.models import AssetFile, AssetRef, AssetType
        from molexp.utils.id import compute_content_hash, generate_asset_id

        source_path = PathLib(uri)
        if not source_path.exists():
            return asset  # Can't store non-existent file

        # Compute content hash
        content_hash = compute_content_hash(source_path)

        # Check if asset already exists
        existing_asset_id = ctx.workspace.find_asset_by_hash(content_hash)

        if existing_asset_id:
            asset_id = existing_asset_id
        else:
            # Create new asset
            asset_id = generate_asset_id()

            # Determine asset type from file extension
            ext = source_path.suffix.lower()
            asset_type_map = {
                ".pdb": AssetType.STRUCTURE,
                ".xyz": AssetType.STRUCTURE,
                ".mol2": AssetType.STRUCTURE,
                ".xtc": AssetType.TRAJECTORY,
                ".dcd": AssetType.TRAJECTORY,
                ".trr": AssetType.TRAJECTORY,
                ".top": AssetType.TOPOLOGY,
                ".itp": AssetType.FORCEFIELD,
                ".png": AssetType.IMAGE,
                ".jpg": AssetType.IMAGE,
                ".csv": AssetType.TABLE,
                ".json": AssetType.TABLE,
                ".log": AssetType.LOG,
            }
            asset_type = asset_type_map.get(ext, AssetType.OTHER)

            asset_model = AssetModel(
                asset_id=asset_id,
                type=asset_type,
                format=ext.lstrip(".") or "unknown",
                created_at=datetime.now(),
                producer_run_id=ctx.run_metadata.run_id,
                size_bytes=source_path.stat().st_size,
                content_hash=content_hash,
                mime_type="",
                tags=[],
                metadata=meta or {},
                files=[
                    AssetFile(
                        path=f"data/{source_path.name}",
                        size=source_path.stat().st_size,
                        hash=content_hash,
                    )
                ],
            )

            ctx.workspace.store_asset(asset_model, source_path)

        # Create AssetRef
        asset_ref = AssetRef(
            asset_id=asset_id,
            role=label or "output",
            producer_run_id=ctx.run_metadata.run_id,
            produced_at=datetime.now(),
        )

        # Update asset_refs.json
        refs = ctx.workspace.get_asset_refs(
            ctx.run_metadata.project_id,
            ctx.run_metadata.experiment_id,
            ctx.run_metadata.run_id,
        )
        if refs:
            refs.outputs.append(asset_ref)
            ctx.workspace.save_asset_refs(
                ctx.run_metadata.project_id,
                ctx.run_metadata.experiment_id,
                ctx.run_metadata.run_id,
                refs,
            )

    return asset
