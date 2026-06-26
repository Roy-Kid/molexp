"""RED tests for ``molexp.workspace.curation`` asset queries.

Pins ``find_asset_by_hash`` (content-addressed lookup composing
``catalog.find_by_content_hash``) and ``aggregate_assets_by_kind``
(composing ``{scope}.assets.query`` and counting by ``kind``).

Both functions are read-only and must NOT trigger a ``catalog.rebuild()``:
imported ``DataAsset`` rows are registered directly in the catalog but are
not written to a scope ``assets.json`` manifest, so a rebuild would drop
them. These tests therefore query immediately after import.

Until ``molexp.workspace.curation`` exists they fail at collection with
``ModuleNotFoundError`` — the intended RED state.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.curation import aggregate_assets_by_kind, find_asset_by_hash

# ── find_asset_by_hash ───────────────────────────────────────────────────────


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


# ── aggregate_assets_by_kind ─────────────────────────────────────────────────


class TestAggregateAssetsByKind:
    def test_counts_workspace_scoped_data_asset(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Agg Lab")
        src = tmp_path / "a.txt"
        src.write_text("aaa")
        ws.data_assets.import_asset("a", src)

        assert aggregate_assets_by_kind(ws) == {"data": 1}

    def test_counts_multiple_data_assets_under_one_kind(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Agg Lab")
        for i in range(2):
            src = tmp_path / f"a{i}.txt"
            src.write_text(f"content-{i}")
            ws.data_assets.import_asset(f"a{i}", src)

        assert aggregate_assets_by_kind(ws) == {"data": 2}

    def test_non_recursive_experiment_scope_sees_no_run_assets(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Agg Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        run = exp.add_run(params={"seed": 0})
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"x": 1})

        # Artifacts are run-scoped; the experiment scope is empty non-recursively.
        assert aggregate_assets_by_kind(exp) == {}

    def test_recursive_experiment_scope_sees_run_assets(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Agg Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        run = exp.add_run(params={"seed": 0})
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"x": 1})

        result = aggregate_assets_by_kind(exp, recursive=True)
        assert result.get("artifact") == 1
        # the run lifecycle auto-creates a "run" log, also visible recursively
        assert "log" in result
