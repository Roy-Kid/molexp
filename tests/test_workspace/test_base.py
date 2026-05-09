"""Tests for :mod:`molexp.workspace.base` atomic-write helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import atomic_write_text


def test_atomic_write_text_round_trips_utf8(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    payload = "# Header\n\nbody — with é, ñ, 中 unicode\n"
    atomic_write_text(target, payload)
    assert target.read_text(encoding="utf-8") == payload


def test_atomic_write_text_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.txt"
    atomic_write_text(target, "ok")
    assert target.exists()
    assert target.read_text() == "ok"


def test_atomic_write_text_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text() == "second"


def test_atomic_write_text_leaves_no_temp_file_on_success(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    atomic_write_text(target, "ok")
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_atomic_write_text_cleans_up_temp_file_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing ``os.replace`` must not leave a stray temp file."""
    import os

    def _boom(_src: str, _dst: str | Path) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", _boom)
    target = tmp_path / "out.txt"
    with pytest.raises(OSError):
        atomic_write_text(target, "ok")
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    # Original file should not have been created.
    assert not target.exists()


def test_atomic_write_text_supports_alternate_encoding(tmp_path: Path) -> None:
    target = tmp_path / "latin.txt"
    payload = "café"
    atomic_write_text(target, payload, encoding="latin-1")
    # Re-reading with latin-1 returns the original; with utf-8 it
    # would be mojibake. Either is fine — this asserts encoding is honored.
    assert target.read_bytes() == payload.encode("latin-1")
