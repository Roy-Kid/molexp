"""DataAsset — user-imported inputs (datasets, models, external files).

Lives at ``{scope}/assets/<asset_id>/payload/``.  ``DataAssetLibrary``
replaces the old ``AssetLibrary`` as the import surface; the catalog
registration side is orchestrated from the scope entity.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..utils import compute_content_hash, generate_asset_id
from .base import Asset, AssetScope

if TYPE_CHECKING:
    from .catalog import AssetCatalog


class DataAsset(Asset):
    """User-imported input.

    ``payload_dir`` points to a directory under the scope's
    ``assets/<asset_id>/`` that contains the raw content.
    """

    kind: Literal["data"] = "data"
    source_path: str
    import_action: Literal["copy", "move", "symlink", "hardlink"] = "copy"
    content_hash: str | None = None

    def payload(self, scope_dir: Path, rel: str = "") -> Path:
        """Return the payload directory (optionally joined with a relative path)."""
        base = self.absolute_path(scope_dir)
        return base / rel if rel else base


class DataAssetLibrary:
    """Import surface for ``DataAsset`` at one scope.

    The library is responsible for:
      * materializing the imported content under ``{scope_dir}/assets/<asset_id>/payload/``
      * constructing the ``DataAsset`` record
      * delegating registration (manifest + catalog) to the scope entity

    ``catalog`` is optional — passing ``None`` leaves callers responsible
    for registration (useful for tests).
    """

    def __init__(
        self,
        scope_dir: Path,
        scope: AssetScope,
        catalog: "AssetCatalog | None" = None,
    ) -> None:
        self.scope_dir = scope_dir
        self.scope = scope
        self.catalog = catalog
        self.root = scope_dir / "assets"

    def import_asset(
        self,
        name: str,
        src: str | Path,
        action: Literal["copy", "move", "symlink", "hardlink"] = "copy",
        meta: dict[str, Any] | None = None,
    ) -> DataAsset:
        """Import a file or directory as a ``DataAsset``."""
        source_path = Path(src).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {src}")

        asset_id = generate_asset_id()
        asset_dir = self.root / asset_id
        payload_dir = asset_dir / "payload"

        self._materialize(source_path, payload_dir, action)

        now = datetime.now()
        rel_path = Path("assets") / asset_id / "payload"
        content_hash = None
        if source_path.is_file() and action in ("copy", "move"):
            content_hash = compute_content_hash(payload_dir)

        asset = DataAsset(
            asset_id=asset_id,
            name=name,
            scope=self.scope,
            path=rel_path,
            created_at=now,
            updated_at=now,
            producer=None,
            tags=meta or {},
            source_path=str(source_path),
            import_action=action,
            content_hash=content_hash,
        )

        asset_dir.mkdir(parents=True, exist_ok=True)
        with open(asset_dir / "asset.json", "w") as f:
            json.dump(asset.model_dump(mode="json"), f, indent=2)

        if self.catalog is not None:
            self.catalog.register(asset)

        return asset

    def list_assets(self) -> list[DataAsset]:
        """List all DataAssets in this scope by scanning the filesystem."""
        if not self.root.exists():
            return []
        out: list[DataAsset] = []
        for asset_dir in self.root.iterdir():
            meta_file = asset_dir / "asset.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    data = json.load(f)
                out.append(DataAsset.model_validate(data))
        return out

    def get(self, name: str) -> DataAsset | None:
        for asset in self.list_assets():
            if asset.name == name:
                return asset
        return None

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _materialize(src: Path, dest: Path, action: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if action == "copy":
            if src.is_file():
                shutil.copy2(src, dest)
            else:
                shutil.copytree(src, dest, dirs_exist_ok=True)
        elif action == "move":
            shutil.move(str(src), str(dest))
        elif action == "symlink":
            dest.symlink_to(src)
        elif action == "hardlink":
            if src.is_file():
                try:
                    os.link(src, dest)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dest)
            else:
                dest.mkdir(parents=True, exist_ok=True)
                for item in src.rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(src)
                        target = dest / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(item, target)
                        except (OSError, NotImplementedError):
                            shutil.copy2(item, target)
        else:
            raise ValueError(f"Unknown import action: {action!r}")
