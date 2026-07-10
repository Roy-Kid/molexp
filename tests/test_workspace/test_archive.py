"""``archive_folder_zip`` — Folder → zip bytes (agent-record-export-03)."""

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


def _legacy_inline_zip(root: Path) -> bytes:
    """Replay server/routes/run.py export_run inline zip construction."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(root).as_posix())
    return buffer.getvalue()


def test_nested_tree_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub" / "deep").mkdir(parents=True)
    (root / "a.txt").write_text("A")
    (root / "sub" / "b.txt").write_text("B")
    (root / "sub" / "deep" / "c.bin").write_bytes(b"\x00\x01")
    folder = _folder_at(root)

    data = archive_folder_zip(folder)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert names == ["a.txt", "sub/b.txt", "sub/deep/c.bin"]
        assert zf.read("a.txt") == b"A"
        assert zf.read("sub/b.txt") == b"B"
        assert zf.read("sub/deep/c.bin") == b"\x00\x01"
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED
            assert not info.filename.endswith("/")


def test_parts_tuple_sort_order(tmp_path: Path) -> None:
    root = tmp_path / "sort"
    (root / "a").mkdir(parents=True)
    (root / "a-b.txt").write_text("dash")
    (root / "a" / "b.txt").write_text("nested")
    folder = _folder_at(root)

    with zipfile.ZipFile(io.BytesIO(archive_folder_zip(folder))) as zf:
        # parts-tuple order: a/b.txt before a-b.txt
        assert zf.namelist() == ["a/b.txt", "a-b.txt"]


def test_empty_and_missing_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive_folder_zip(_folder_at(empty)))) as zf:
        assert zf.namelist() == []

    missing = tmp_path / "no-such"
    folder = Folder(
        name="no-such",
        kind="test.folder",
        root_path=str(tmp_path),
        fs=LocalFileSystem(),
    )
    assert not missing.exists()
    with zipfile.ZipFile(io.BytesIO(archive_folder_zip(folder))) as zf:
        assert zf.namelist() == []
    assert not missing.exists(), "resolve()-based archive must not mkdir"


def test_deterministic_bytes(tmp_path: Path) -> None:
    root = tmp_path / "det"
    root.mkdir()
    (root / "x.txt").write_text("same")
    folder = _folder_at(root)
    assert archive_folder_zip(folder) == archive_folder_zip(folder)


def test_namelist_parity_with_legacy_inline(tmp_path: Path) -> None:
    root = tmp_path / "parity"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("A")
    (root / "sub" / "b.txt").write_text("B")
    folder = _folder_at(root)

    ours = archive_folder_zip(folder)
    legacy = _legacy_inline_zip(root)
    with zipfile.ZipFile(io.BytesIO(ours)) as z1, zipfile.ZipFile(io.BytesIO(legacy)) as z2:
        assert z1.namelist() == z2.namelist()
        for name in z1.namelist():
            assert z1.read(name) == z2.read(name)
