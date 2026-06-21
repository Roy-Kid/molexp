"""Tests for the Note + Reference concept types (okf-07-01).

Notes and references are ``Folder`` subclasses (concept types, mountable
anywhere). A note's body lives in ``index.md`` and its citations are markdown
links (resolved by ``out_edges``); a reference's structured bib fields live in
``meta.yaml`` (``ReferenceMeta``). Each ref/note is its own Concept directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.knowledge import (
    Folder,
    Library,
    Note,
    Reference,
    ReferenceMeta,
)

# ── ReferenceMeta (ac-002) ───────────────────────────────────────────────────


def test_reference_meta_defaults_and_round_trip() -> None:
    m = ReferenceMeta(title="Deep Learning", authors=("LeCun", "Bengio"), year=2015, doi="10.1/x")
    assert m.type == "reference"
    assert m.source == "manual"
    back = ReferenceMeta.from_yaml(m.to_yaml())
    assert isinstance(back, ReferenceMeta)
    assert back.title == "Deep Learning"
    assert back.authors == ("LeCun", "Bengio")
    assert back.year == 2015
    assert back.doi == "10.1/x"


def test_reference_meta_frozen() -> None:
    m = ReferenceMeta()
    with pytest.raises(ValidationError):
        m.title = "x"  # type: ignore[misc]


# ── registry returns the right subclass (ac-003) ─────────────────────────────


def test_registry_returns_note_and_reference(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    note = root.add_folder("idea", concept_type="note")
    ref = root.add_folder("smith2024", concept_type="reference")
    assert isinstance(note, Note)
    assert isinstance(ref, Reference)
    assert note.read_meta().type == "note"
    assert ref.read_meta().type == "reference"


# ── Note body + cite (ac-004) ────────────────────────────────────────────────


def test_note_body_and_cite_round_trip(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    note = root.add_folder("idea", concept_type="note")
    ref = root.add_folder("smith2024", concept_type="reference")

    note.set_body("# Idea\n\nbuilds on prior work\n")
    assert "builds on prior work" in note.body()

    note.cite(ref)
    assert "smith2024" in note.read_index()  # citation is a markdown link
    assert Path(ref.resolve()) in {Path(p) for p in note.out_edges()}


# ── Reference typed meta (ac-005) ────────────────────────────────────────────


def test_reference_typed_meta_and_citation(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    ref = root.add_folder("smith2024", concept_type="reference")

    ref.write_ref_meta(ReferenceMeta(title="T", doi="10.1/x", year=2024))
    got = ref.read_ref_meta()
    assert isinstance(got, ReferenceMeta)
    assert got.title == "T"
    assert got.doi == "10.1/x"
    assert got.year == 2024

    ref.set_citation("Smith et al. 2024")
    assert ref.citation() == "Smith et al. 2024"


# ── append_link extraction keeps Library.link (ac-006) ───────────────────────


def test_library_link_still_round_trips(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    a = root.add_folder("a")
    b = root.add_folder("b")
    lib = Library(tmp_path)
    lib.link(a, b)
    assert Path(b.resolve()) in {Path(p) for p in a.out_edges()}


# ── Library filtered views (ac-007) ──────────────────────────────────────────


def test_library_references_and_notes_views(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    root.add_folder("n1", concept_type="note")
    root.add_folder("r1", concept_type="reference")
    root.add_folder("plain")
    lib = Library(tmp_path)

    assert [r.name for r in lib.references()] == ["r1"]
    assert [n.name for n in lib.notes()] == ["n1"]
    assert all(isinstance(r, Reference) for r in lib.references())
    assert all(isinstance(n, Note) for n in lib.notes())
