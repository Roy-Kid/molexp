"""Workspace path toolkit (``ws.wp`` / ``me.wp``) for layout fixes."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import Workspace, mv, validate_workspace


def _ws(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab")
    ws.materialize()
    return ws


class TestWorkspacePaths:
    def test_mv_moves_stray_under_assets(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        root = Path(ws.resolve())
        stray = root / "leftover-run-output"
        stray.mkdir()
        (stray / "traj.pt").write_bytes(b"x")

        report = validate_workspace(root)
        assert "layout.stray" in {v.rule for v in report.errors}

        ws.wp.mkdir("projects/demo/assets")
        dest = ws.wp.mv("leftover-run-output", "projects/demo/assets/nve-verify")
        assert Path(dest).name == "nve-verify"
        assert not stray.exists()
        assert Path(dest).is_dir()
        assert (Path(dest) / "traj.pt").is_file()
        # Stray at root is gone — layout.stray for that path clears.
        again = validate_workspace(root)
        assert "leftover-run-output" not in {
            v.path for v in again.errors if v.rule == "layout.stray"
        }

    def test_mv_into_existing_dir_keeps_basename(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        root = Path(ws.resolve())
        (root / "blob").write_text("hi")
        (root / "projects").mkdir(exist_ok=True)

        dest = ws.wp.mv("blob", "projects")
        assert Path(dest).as_posix().endswith("projects/blob")
        assert (root / "projects" / "blob").read_text() == "hi"
        assert not (root / "blob").exists()

    def test_mv_refuses_escape(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        with pytest.raises(ValueError, match="outside workspace"):
            ws.wp.mv(".", "/tmp/escape")

    def test_free_function_form(self, tmp_path: Path) -> None:
        import molexp as me

        ws = _ws(tmp_path)
        root = Path(ws.resolve())
        (root / "a").write_text("1")
        dest = me.wp.mv(ws, "a", "b")
        assert Path(dest).name == "b"
        assert (root / "b").read_text() == "1"

    def test_ls_mkdir_rm_cp(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        ws.wp.mkdir("stash/nested")
        (Path(ws.resolve()) / "stash" / "nested" / "f.txt").write_text("z")
        names = ws.wp.ls("stash")
        assert "nested" in names
        ws.wp.cp("stash/nested/f.txt", "stash/copy.txt")
        assert (Path(ws.resolve()) / "stash" / "copy.txt").read_text() == "z"
        ws.wp.rm("stash/copy.txt")
        assert not (Path(ws.resolve()) / "stash" / "copy.txt").exists()
        ws.wp.rm("stash", recursive=True)
        assert not (Path(ws.resolve()) / "stash").exists()

    def test_module_level_mv_alias(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        (Path(ws.resolve()) / "x").write_text("y")
        mv(ws, "x", "z")
        assert (Path(ws.resolve()) / "z").read_text() == "y"
