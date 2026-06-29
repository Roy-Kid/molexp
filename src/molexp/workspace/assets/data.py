"""DataAsset — user-imported inputs (datasets, models, external files).

Lives at ``{scope}/assets/<asset_id>/payload/``.  ``DataAssetLibrary``
replaces the old ``AssetLibrary`` as the import surface; each imported
asset is recorded authoritatively as ``{scope}/assets/<asset_id>/asset.json``.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from ..utils import compute_content_hash, generate_asset_id
from .base import Asset, AssetScope, Producer

ImportAction = Literal["copy", "move", "symlink", "hardlink", "reference"]


class DataAsset(Asset):
    """User-imported input.

    ``payload_dir`` points to a directory under the scope's
    ``assets/<asset_id>/`` that contains the raw content.
    """

    kind: Literal["data"] = "data"
    source_path: str
    import_action: ImportAction = "copy"

    def payload(self, scope_dir: Path, rel: str = "") -> Path:
        """Return the payload directory (optionally joined with a relative path)."""
        base = self.absolute_path(scope_dir)
        return base / rel if rel else base


class DataAssetLibrary:
    """Import surface for ``DataAsset`` at one scope.

    The library is responsible for:
      * materializing the imported content under ``{scope_dir}/assets/<asset_id>/payload/``
      * constructing the ``DataAsset`` record
      * writing the authoritative ``{scope_dir}/assets/<asset_id>/asset.json`` record
    """

    def __init__(
        self,
        scope_dir: str | PathLike[str],
        scope: AssetScope,
    ) -> None:
        # Coerce to pathlib.Path — DataAssetLibrary does genuine local I/O
        # (shutil.copy2, os.link, symlink_to); callers can pass molexp.Path
        # or str.  Remote DataAsset storage is not supported by this class.
        self.scope_dir = Path(scope_dir)
        self.scope = scope
        self.root = self.scope_dir / "assets"

    def import_asset(
        self,
        name: str,
        src: str | Path,
        action: ImportAction = "copy",
        meta: dict[str, Any] | None = None,
        *,
        consumed: list[Asset] | tuple[Asset, ...] | None = None,
    ) -> DataAsset:
        """Import a file or directory as a ``DataAsset``.

        Args:
            name: Asset display name.
            src: Path to import.
            action: How to materialize (``copy`` / ``move`` / ``symlink``
                / ``hardlink``).
            meta: Free-form tags persisted with the asset.
            consumed: Optional upstream assets used to build this one.
                Their ``asset_id``s are recorded in
                :attr:`Producer.inputs` so lineage queries can trace
                back through derived data products.
        """
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
        if action in ("copy", "move") and payload_dir.exists():
            content_hash = compute_content_hash(payload_dir)

        producer: Producer | None = None
        if consumed:
            producer = Producer(inputs=tuple(a.asset_id for a in consumed))

        asset = DataAsset(
            asset_id=asset_id,
            name=name,
            scope=self.scope,
            path=rel_path,
            created_at=now,
            updated_at=now,
            producer=producer,
            tags=meta or {},
            source_path=str(source_path),
            import_action=action,
            content_hash=content_hash,
        )

        asset_dir.mkdir(parents=True, exist_ok=True)
        with open(asset_dir / "asset.json", "w") as f:  # noqa: PTH123
            json.dump(asset.model_dump(mode="json"), f, indent=2)

        return asset

    def register_in_place(
        self,
        name: str,
        src: str | Path,
        meta: dict[str, Any] | None = None,
    ) -> DataAsset:
        """Register a file already inside the scope as a ``DataAsset``.

        Unlike :meth:`import_asset`, nothing is copied or moved: the asset's
        ``path`` points at ``src`` where it already lives (relative to the
        scope directory). This keeps the original filename, so a same-stem
        sidecar (``qm9.tar.bz2`` ↔ ``qm9.py``) stays a real sibling of the
        resolved path. The reference is recorded authoritatively as
        ``assets/<id>/asset.json`` (record only — no ``payload/`` directory,
        since the file stays where it already lives).

        Args:
            name: Asset display name.
            src: Path to a file/directory that must already live under the
                scope directory.
            meta: Free-form tags persisted with the asset.

        Returns:
            The registered :class:`DataAsset`.

        Raises:
            FileNotFoundError: ``src`` does not exist.
            ValueError: ``src`` is not under the scope directory.
        """
        source_path = Path(src).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {src}")

        # Must live under the scope dir — the asset path is scope-relative for
        # portability; ``relative_to`` raises ValueError otherwise.
        rel_path = source_path.relative_to(self.scope_dir.resolve())

        now = datetime.now()
        content_hash = compute_content_hash(source_path)

        asset = DataAsset(
            asset_id=generate_asset_id(),
            name=name,
            scope=self.scope,
            path=rel_path,
            created_at=now,
            updated_at=now,
            tags=meta or {},
            source_path=str(source_path),
            import_action="reference",
            content_hash=content_hash,
        )

        # Authoritative on-disk record (no payload/ dir — the file stays in
        # place); the scanner reads assets/<id>/asset.json.
        asset_dir = self.root / asset.asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        with open(asset_dir / "asset.json", "w") as f:  # noqa: PTH123
            json.dump(asset.model_dump(mode="json"), f, indent=2)

        return asset

    def list_assets(self) -> list[DataAsset]:
        """List all DataAssets in this scope by scanning the filesystem."""
        if not self.root.exists():
            return []
        out: list[DataAsset] = []
        for asset_dir in self.root.iterdir():
            meta_file = asset_dir / "asset.json"
            if meta_file.exists():
                with open(meta_file) as f:  # noqa: PTH123
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
