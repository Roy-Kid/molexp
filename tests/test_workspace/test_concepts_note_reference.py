"""Tests for the OKF Note + Reference concept types on ``workspace.Folder`` (wsokf-05).

Mirrors ``tests/test_knowledge/test_references.py`` against the workspace surface.
Notes and references are ``Folder`` subclasses (concept types, mountable
anywhere). A note's body lives in ``index.md`` and its citations are markdown
links (resolved by ``out_edges``); a reference's structured bib fields live in
``meta.yaml`` (``ReferenceMeta``). Each ref/note is its own Concept directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from molexp.workspace import (
    Note,
    ReferenceConcept,
    ReferenceMeta,
)
from molexp.workspace.folder import Folder, append_link, concept_from_dir

# ── ReferenceMeta (ac-001) ───────────────────────────────────────────────────


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


def test_reference_meta_extra_allow_round_trip() -> None:
    back = cast("ReferenceMeta", ReferenceMeta.from_yaml("type: reference\ntitle: T\nx: kept\n"))
    assert back.title == "T"
    # extra="allow" preserves unknown keys verbatim
    assert back.__pydantic_extra__ is not None
    assert back.__pydantic_extra__["x"] == "kept"


# ── registry returns the right subclass (ac-002) ─────────────────────────────


def _mount[F: Folder](parent: Folder, child: F) -> F:
    """Mount a self-parented concept child via the generic five-verb CRUD."""
    return cast("F", parent.add_folder(child))


def test_registry_returns_note_and_reference(tmp_path: Path) -> None:
    root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
    root.materialize()
    note = _mount(root, Note(parent=root, name="idea"))
    ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))
    assert isinstance(note, Note)
    assert isinstance(ref, ReferenceConcept)
    assert note.read_meta()["type"] == "note.note"
    assert ref.read_meta()["type"] == "reference.reference"


def test_concept_from_dir_rebuilds_note_and_reference(tmp_path: Path) -> None:
    root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
    root.materialize()
    note = _mount(root, Note(parent=root, name="idea"))
    ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

    rebuilt_note = concept_from_dir(note.resolve(), root)
    rebuilt_ref = concept_from_dir(ref.resolve(), root)
    assert isinstance(rebuilt_note, Note)
    assert isinstance(rebuilt_ref, ReferenceConcept)


# ── Note body + cite (ac-003 / ac-005) ───────────────────────────────────────


def test_note_body_and_cite_round_trip(tmp_path: Path) -> None:
    root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
    root.materialize()
    note = _mount(root, Note(parent=root, name="idea"))
    ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

    note.set_body("# Idea\n\nbuilds on prior work\n")
    assert "builds on prior work" in note.body()

    note.cite(ref)
    assert "smith2024" in note.read_index()  # citation is a markdown link
    assert Path(ref.resolve()) in {Path(p) for p in note.out_edges()}


def test_append_link_module_level_on_folder(tmp_path: Path) -> None:
    root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
    root.materialize()
    a = _mount(root, Note(parent=root, name="a"))
    b = _mount(root, Note(parent=root, name="b"))
    append_link(a, b)
    assert Path(b.resolve()) in {Path(p) for p in a.out_edges()}
    # the edge is markdown, never smuggled into meta.yaml
    assert "b" not in (Path(a.resolve()) / "meta.yaml").read_text(encoding="utf-8")


# ── Reference typed meta + citation (ac-004) ─────────────────────────────────


def test_reference_typed_meta_and_citation(tmp_path: Path) -> None:
    root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
    root.materialize()
    ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

    ref.write_ref_meta(ReferenceMeta(title="T", doi="10.1/x", year=2024))
    got = ref.read_ref_meta()
    assert isinstance(got, ReferenceMeta)
    assert got.title == "T"
    assert got.doi == "10.1/x"
    assert got.year == 2024

    ref.set_citation("Smith et al. 2024")
    assert ref.citation() == "Smith et al. 2024"


# ── OKF concept exports (ac-008 / ac-009) ────────────────────────────────────


def test_okf_concepts_exported() -> None:
    import molexp.workspace as workspace

    # OKF concepts exported under unambiguous names
    assert "Note" in workspace.__all__
    assert "ReferenceConcept" in workspace.__all__
    assert "ReferenceMeta" in workspace.__all__

    # The legacy library surface is gone (wsokf-11).
    for legacy in ("Library", "LibraryIndex", "NoteEntry", "NoteAsset", "ReferenceStore"):
        assert legacy not in workspace.__all__
        assert not hasattr(workspace, legacy)
