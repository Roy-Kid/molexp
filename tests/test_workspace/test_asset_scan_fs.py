"""Asset scan works through an injected FileSystem (remote parity)."""

from __future__ import annotations

import json
from pathlib import Path

from molexp.workspace.assets.scan import get_asset, scan_assets
from molexp.workspace.fs_local import LocalFileSystem


def _write_manifest(scope: Path, assets: dict) -> None:
    scope.mkdir(parents=True, exist_ok=True)
    (scope / "assets.json").write_text(
        json.dumps({"schema_version": 1, "assets": assets}),
        encoding="utf-8",
    )


def test_scan_assets_via_fs_matches_path(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    # minimal asset-shaped record
    asset = {
        "asset_id": "aaaaaaaaaaaa",
        "name": "out",
        "kind": "artifact",
        "scope": {"kind": "workspace", "ids": []},
        "path": "out.dat",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "content_hash": "",
        "tags": {},
        "producer": None,
    }
    _write_manifest(root, {"aaaaaaaaaaaa": asset})

    via_path = scan_assets(root)
    via_fs = scan_assets(str(root), fs=LocalFileSystem())
    assert len(via_path) == 1
    assert len(via_fs) == 1
    assert via_path[0].asset_id == via_fs[0].asset_id == "aaaaaaaaaaaa"
    assert get_asset(str(root), "aaaaaaaaaaaa", fs=LocalFileSystem()) is not None
