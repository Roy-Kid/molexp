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


# ── okf-01-01 back-compat: re-exports must be the SAME function objects ──
# After promoting the storage primitives to molexp.atomicio / molexp.ids,
# every legacy import path must resolve to the identical object so existing
# call sites and monkeypatch targets keep working.


def test_workspace_base_atomic_writers_are_atomicio_objects() -> None:
    import molexp.atomicio as atomicio
    import molexp.workspace.base as base

    assert base.atomic_write_json is atomicio.atomic_write_json
    assert base.atomic_write_text is atomicio.atomic_write_text
    assert base._atomic_write_json is atomicio.atomic_write_json


def test_workspace_top_level_atomic_writers_are_atomicio_objects() -> None:
    import molexp.atomicio as atomicio
    import molexp.workspace as ws

    assert ws.atomic_write_json is atomicio.atomic_write_json
    assert ws.atomic_write_text is atomicio.atomic_write_text


def test_workspace_utils_id_primitives_are_ids_objects() -> None:
    import molexp.ids as ids
    import molexp.workspace.utils as utils

    assert utils.slugify is ids.slugify
    assert utils.generate_id is ids.generate_id
    assert utils.generate_asset_id is ids.generate_asset_id
    assert utils.compute_content_hash is ids.compute_content_hash
    # Run-domain derivations stay defined in workspace.utils.
    assert callable(utils.derive_run_id)
    assert callable(utils.derive_execution_id)


def test_workspace_file_lock_symbols_are_atomicio_objects() -> None:
    import molexp.atomicio as atomicio
    import molexp.workspace._file_lock as fl

    assert fl.file_lock is atomicio.file_lock
    assert fl.FileLockTimeoutError is atomicio.FileLockTimeoutError
