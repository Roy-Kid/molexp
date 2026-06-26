"""Tests for ``molexp.workspace.curation`` reorganization helpers.

Pins ``move_run`` (relocate a Run to another Experiment, including the
destination-collision guard), ``rehome_asset`` (re-import a DataAsset payload
into another scope, preserving its content hash) and ``delete_folder``.

``reslug`` (entity rename) is intentionally deferred: renaming an entity's id
must rewrite its authoritative metadata file and re-home every asset cataloged
under the old scope (and a Run's id is embedded in its execution ids), which is
a focused follow-up rather than a thin ``move_to`` compose.

The assertions describe the *correct end state* the curation layer must
produce — e.g. after ``move_run`` the run is discoverable via the target
experiment's ``list_runs()`` (which scans the ``runs/`` container on disk)
and gone from the source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import FolderMoveCollisionError, Workspace
from molexp.workspace.curation import delete_folder, move_run, rehome_asset

# ── move_run ─────────────────────────────────────────────────────────────────


class TestMoveRun:
    def test_run_relocates_to_target_experiment(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Reorg Lab")
        proj = ws.add_project("proj")
        source_exp = proj.add_experiment("source-exp", params={})
        target_exp = proj.add_experiment("target-exp", params={})
        run = source_exp.add_run(params={"seed": 0})
        old_dir = Path(str(run.run_dir))
        assert old_dir.exists()

        move_run(run, target_exp)

        assert run.id in [r.id for r in target_exp.list_runs()]
        assert run.id not in [r.id for r in source_exp.list_runs()]
        assert not old_dir.exists()

    def test_collision_propagates(self, tmp_path: Path) -> None:
        # The target experiment already holds a run at the same id, so the
        # underlying move_to must refuse rather than clobber it. The typed
        # collision error is propagated verbatim, never caught-and-continued.
        ws = Workspace(root=tmp_path / "lab", name="Collision Lab")
        proj = ws.add_project("proj")
        source_exp = proj.add_experiment("source-exp", params={})
        target_exp = proj.add_experiment("target-exp", params={})
        run = source_exp.add_run(params={"seed": 0}, id="shared")
        target_exp.add_run(params={"seed": 1}, id="shared")

        with pytest.raises(FolderMoveCollisionError):
            move_run(run, target_exp)


# ── rehome_asset ─────────────────────────────────────────────────────────────


class TestRehomeAsset:
    @pytest.mark.parametrize("action", ["copy", "move"])
    def test_content_hash_preserved(self, tmp_path: Path, action: str) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Rehome Lab")
        target = ws.add_project("dest")

        payload_src = tmp_path / "payload"
        payload_src.mkdir()
        (payload_src / "data.bin").write_text("payload-bytes")
        asset = ws.data_assets.import_asset("dataset", payload_src, action="copy")
        original_hash = asset.content_hash
        assert original_hash is not None
        assert original_hash.startswith("sha256:")

        rehomed = rehome_asset(asset, source=ws, target=target, action=action)
        assert rehomed.content_hash == original_hash
        assert rehomed.name == asset.name


# ── delete_folder ────────────────────────────────────────────────────────────


class TestDeleteFolder:
    def test_removes_folder_and_drops_from_listing(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Delete Lab")
        proj = ws.add_project("doomed")
        proj_dir = Path(str(proj.project_dir))
        assert proj_dir.exists()

        delete_folder(proj)

        assert not proj_dir.exists()
        assert "doomed" not in [p.id for p in ws.list_projects()]
