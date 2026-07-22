"""Tests for ``molexp.workspace.curation.query`` asset queries.

Pins ``find_asset_by_hash`` (content-addressed lookup over the authoritative
manifests) and ``aggregate_assets_by_kind`` (counts a scope's assets by
``kind``, honoring the ``recursive`` flag). Both are read-only compositions
over ``scan.scan_assets`` / per-scope ``AssetsView`` — no ``catalog.rebuild()``.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.curation import aggregate_assets_by_kind, find_asset_by_hash


class TestFindAssetByHash:
    def test_returns_imported_asset(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Query Lab")
        src = tmp_path / "input.txt"
        src.write_text("hello world")
        asset = ws.data_assets.import_asset("greeting", src)

        found = find_asset_by_hash(ws, asset.content_hash)
        assert found is not None
        assert found.asset_id == asset.asset_id
        assert found.content_hash == asset.content_hash

    def test_unknown_hash_returns_none(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Query Lab")
        ws.materialize()
        assert find_asset_by_hash(ws, "sha256:deadbeef") is None


class TestAggregateAssetsByKind:
    def test_counts_in_scope_asset_by_kind(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Agg Lab")
        src = tmp_path / "a.txt"
        src.write_text("aaa")
        ws.data_assets.import_asset("a", src)

        assert aggregate_assets_by_kind(ws) == {"data": 1}

    def test_non_recursive_scope_excludes_sub_scope_assets(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Agg Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        run = exp.add_run(params={"seed": 0})
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"x": 1})

        # Artifacts are run-scoped; the experiment scope is empty non-recursively.
        assert aggregate_assets_by_kind(exp) == {}

    def test_recursive_scope_includes_sub_scope_assets(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Agg Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        run = exp.add_run(params={"seed": 0})
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"x": 1})

        result = aggregate_assets_by_kind(exp, recursive=True)
        assert result.get("artifact") == 1
        # the run lifecycle auto-creates a "run" log, also visible recursively
        assert "log" in result
