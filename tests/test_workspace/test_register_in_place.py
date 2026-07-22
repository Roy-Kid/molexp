"""``DataAssetLibrary.register_in_place`` — reference an in-tree file as a DataAsset.

A dataset already living in the workspace tree (e.g. a QM9 download sitting next
to its ``qm9.py`` reader sidecar) is recorded into the asset index *without*
being copied into an opaque ``assets/<id>/payload`` directory. The asset's
``path`` keeps the original name, so a same-stem sidecar stays a real sibling of
the resolved path.
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace


class TestDataAssetLibraryRegisterInPlace:
    def test_keeps_original_name_and_sidebar_sibling(self, tmp_path):
        ws = Workspace(root=tmp_path, name="T")
        dataset = tmp_path / "qm9.tar.bz2"
        dataset.write_bytes(b"data")
        sidecar = tmp_path / "qm9.py"
        sidecar.write_text("x = 1", encoding="utf-8")

        asset = ws.data_assets.register_in_place(name="qm9.tar.bz2", src=dataset)

        # Path is workspace-relative and keeps the original name (not "payload").
        assert str(asset.path) == "qm9.tar.bz2"
        # Resolves back to the in-place file — nothing was copied.
        assert asset.absolute_path(tmp_path) == dataset
        # Recorded authoritatively as assets/<id>/asset.json.
        from molexp.workspace.assets import scan

        assert scan.get_asset(ws.root, asset.asset_id) is not None
        # The sidecar is a real sibling of the resolved dataset path.
        assert (asset.absolute_path(tmp_path).parent / "qm9.py").exists()

    def test_records_reference_action_and_content_hash(self, tmp_path):
        ws = Workspace(root=tmp_path, name="T")
        f = tmp_path / "d.bin"
        f.write_bytes(b"abc")

        asset = ws.data_assets.register_in_place(name="d.bin", src=f)

        assert asset.import_action == "reference"
        assert asset.content_hash is not None
        assert asset.content_hash.startswith("sha256:")

    def test_rejects_source_outside_the_scope(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        ws = Workspace(root=root, name="T")
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"x")

        with pytest.raises(ValueError):
            ws.data_assets.register_in_place(name="outside.bin", src=outside)

    def test_rejects_missing_source(self, tmp_path):
        ws = Workspace(root=tmp_path, name="T")
        with pytest.raises(FileNotFoundError):
            ws.data_assets.register_in_place(name="ghost", src=tmp_path / "ghost.bin")
