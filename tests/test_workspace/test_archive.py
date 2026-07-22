"""``archive_folder_zip`` — Folder directory → deterministic zip bytes.

The single workspace-layer zip archiver (agent-record-export-03), shared by
agent export and server ``export_run``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from molexp.workspace.archive import archive_folder_zip
from molexp.workspace.folder import Folder
from molexp.workspace.fs_local import LocalFileSystem


def _folder_at(path: Path) -> Folder:
    """Folder whose resolve() is *path* (root_path = parent, name = basename)."""
    path.mkdir(parents=True, exist_ok=True)
    return Folder(
        name=path.name,
        kind="test.folder",
        root_path=str(path.parent),
        fs=LocalFileSystem(),
    )


class TestArchiveFolderZip:
    def test_nested_tree_round_trips_names_and_bytes_deflated(self, tmp_path: Path) -> None:
        root = tmp_path / "tree"
        (root / "sub" / "deep").mkdir(parents=True)
        (root / "a.txt").write_text("A")
        (root / "sub" / "b.txt").write_text("B")
        (root / "sub" / "deep" / "c.bin").write_bytes(b"\x00\x01")

        with zipfile.ZipFile(io.BytesIO(archive_folder_zip(_folder_at(root)))) as zf:
            assert zf.namelist() == ["a.txt", "sub/b.txt", "sub/deep/c.bin"]
            assert zf.read("a.txt") == b"A"
            assert zf.read("sub/b.txt") == b"B"
            assert zf.read("sub/deep/c.bin") == b"\x00\x01"
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_DEFLATED
                assert not info.filename.endswith("/")

    def test_entries_sorted_by_path_parts_not_raw_string(self, tmp_path: Path) -> None:
        # parts-tuple order puts a/b.txt before a-b.txt; raw-string sort would not.
        root = tmp_path / "sort"
        (root / "a").mkdir(parents=True)
        (root / "a-b.txt").write_text("dash")
        (root / "a" / "b.txt").write_text("nested")

        with zipfile.ZipFile(io.BytesIO(archive_folder_zip(_folder_at(root)))) as zf:
            assert zf.namelist() == ["a/b.txt", "a-b.txt"]

    def test_empty_dir_yields_empty_zip(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive_folder_zip(_folder_at(empty)))) as zf:
            assert zf.namelist() == []

    def test_missing_dir_yields_empty_zip_without_mkdir(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such"
        folder = Folder(
            name="no-such",
            kind="test.folder",
            root_path=str(tmp_path),
            fs=LocalFileSystem(),
        )
        with zipfile.ZipFile(io.BytesIO(archive_folder_zip(folder))) as zf:
            assert zf.namelist() == []
        assert not missing.exists(), "resolve()-based archive must not mkdir"

    def test_bytes_are_deterministic(self, tmp_path: Path) -> None:
        root = tmp_path / "det"
        root.mkdir()
        (root / "x.txt").write_text("same")
        folder = _folder_at(root)
        assert archive_folder_zip(folder) == archive_folder_zip(folder)
