"""Tests for sidecar-backed dataset preview.

Host owns discovery + frame cap. The molpy plugin owns reader types.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from molpy.io import BaseTrajectoryReader

from molexp.server.preview import (
    AmbiguousReaderError,
    NoReaderInSidecarError,
    PreviewReaderError,
    PreviewSidecarNotFoundError,
    frames_to_extxyz,
    load_preview,
    preview_frames,
    resolve_sidecar,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "fake_sidecar.py"

_ZERO_READER_SRC = "VALUE = 42\n"

_TWO_READER_SRC = """
from molpy.io import BaseTrajectoryReader
from molpy import Frame


class ReaderA(BaseTrajectoryReader):
    def __init__(self, fpath):
        super().__init__(fpath, must_exist=False)

    def read_frame(self, i):
        return Frame()

    @property
    def n_frames(self):
        return 1


class ReaderB(BaseTrajectoryReader):
    def __init__(self, fpath):
        super().__init__(fpath, must_exist=False)

    def read_frame(self, i):
        return Frame()

    @property
    def n_frames(self):
        return 1
"""
_BROKEN_SRC = "raise RuntimeError('sidecar boom')\n"


def _make_sidecar_dataset(
    dir_path: Path,
    *,
    stem: str = "qm9",
    ext: str = ".bin",
    sidecar_src: str | None = None,
) -> Path:
    dataset = dir_path / f"{stem}{ext}"
    dataset.write_bytes(b"fake dataset content")
    sidecar = dir_path / f"{stem}.py"
    sidecar.write_text(
        _FIXTURE.read_text(encoding="utf-8") if sidecar_src is None else sidecar_src,
        encoding="utf-8",
    )
    return dataset


class TestSidecarPreview:
    def test_resolve_does_not_import_the_sidecar(self, tmp_path, monkeypatch):
        sentinel = tmp_path / "import.sentinel"
        monkeypatch.setenv("MOLEXP_TEST_IMPORT_SENTINEL", str(sentinel))

        dataset = _make_sidecar_dataset(tmp_path)
        info = resolve_sidecar(dataset)

        assert info is not None
        assert info.sidecar_path == tmp_path / "qm9.py"
        assert not sentinel.exists(), "discovery must not execute the sidecar module body"

    def test_load_runs_body_but_not_main_guard(self, tmp_path, monkeypatch):
        import_sentinel = tmp_path / "import.sentinel"
        main_sentinel = tmp_path / "main.sentinel"
        monkeypatch.setenv("MOLEXP_TEST_IMPORT_SENTINEL", str(import_sentinel))
        monkeypatch.setenv("MOLEXP_TEST_MAIN_SENTINEL", str(main_sentinel))

        dataset = _make_sidecar_dataset(tmp_path)
        reader = load_preview(dataset)

        assert isinstance(reader, BaseTrajectoryReader)
        assert import_sentinel.exists(), "explicit load must execute the module body"
        assert not main_sentinel.exists(), "explicit load must not run the __main__ guard"

    def test_load_zero_readers_raises_no_reader(self, tmp_path):
        dataset = _make_sidecar_dataset(tmp_path, sidecar_src=_ZERO_READER_SRC)
        with pytest.raises(NoReaderInSidecarError):
            load_preview(dataset)

    def test_load_two_readers_raises_ambiguous(self, tmp_path):
        dataset = _make_sidecar_dataset(tmp_path, sidecar_src=_TWO_READER_SRC)
        with pytest.raises(AmbiguousReaderError):
            load_preview(dataset)

    def test_load_broken_sidecar_raises_reader_error(self, tmp_path):
        dataset = _make_sidecar_dataset(tmp_path, sidecar_src=_BROKEN_SRC)
        with pytest.raises(PreviewReaderError):
            load_preview(dataset)

    def test_load_missing_sidecar_raises_not_found(self, tmp_path):
        dataset = tmp_path / "plain.bin"
        dataset.write_bytes(b"x")
        with pytest.raises(PreviewSidecarNotFoundError):
            load_preview(dataset)

    def test_preview_frames_caps_at_host_limit(self, tmp_path):
        dataset = _make_sidecar_dataset(tmp_path)
        assert len(preview_frames(dataset, limit=3)) == 3

    def test_frames_to_extxyz_uses_molpy_writer(self, tmp_path):
        dataset = _make_sidecar_dataset(tmp_path)
        xyz = frames_to_extxyz(preview_frames(dataset, limit=2)).decode()
        assert "C" in xyz
        assert "O" in xyz
