"""Tests for the OKF ``Note`` + ``ReferenceConcept`` concept types (wsokf-05).

Notes and references are ``Folder`` subclasses (concept types, mountable
anywhere). A note's body lives in ``index.md`` and its citations are markdown
links (resolved by ``out_edges``); a reference's structured bib fields live in
``meta.yaml`` (``ReferenceMeta``). Each ref/note is its own Concept directory,
reconstructed from ``meta.yaml`` ``type`` via the shared concept-type registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from molexp.workspace import (
    Note,
    ReferenceConcept,
    ReferenceMeta,
)
from molexp.workspace.folder import Folder, concept_from_dir


def _mount[F: Folder](parent: Folder, child: F) -> F:
    """Mount a self-parented concept child via the generic five-verb CRUD."""
    return cast("F", parent.add_folder(child))


class TestReferenceMeta:
    """The typed bib payload (``reference_meta``)."""

    def test_yaml_round_trip_preserves_bib_fields(self) -> None:
        m = ReferenceMeta(
            title="Deep Learning", authors=("LeCun", "Bengio"), year=2015, doi="10.1/x"
        )
        assert m.type == "reference"
        assert m.source == "manual"
        back = ReferenceMeta.from_yaml(m.to_yaml())
        assert isinstance(back, ReferenceMeta)
        assert back.title == "Deep Learning"
        assert back.authors == ("LeCun", "Bengio")
        assert back.year == 2015
        assert back.doi == "10.1/x"


class TestConceptRegistry:
    """``@concept_type`` registration + ``concept_from_dir`` reconstruction."""

    def test_mount_returns_typed_note_and_reference(self, tmp_path: Path) -> None:
        root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
        root.materialize()
        note = _mount(root, Note(parent=root, name="idea"))
        ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))
        assert isinstance(note, Note)
        assert isinstance(ref, ReferenceConcept)
        assert note.read_meta()["type"] == "note.note"
        assert ref.read_meta()["type"] == "reference.reference"

    def test_concept_from_dir_rebuilds_typed_subclass(self, tmp_path: Path) -> None:
        root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
        root.materialize()
        note = _mount(root, Note(parent=root, name="idea"))
        ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

        assert isinstance(concept_from_dir(note.resolve(), root), Note)
        assert isinstance(concept_from_dir(ref.resolve(), root), ReferenceConcept)


class TestNote:
    """``Note`` body + citation edges."""

    def test_body_and_cite_round_trip(self, tmp_path: Path) -> None:
        root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
        root.materialize()
        note = _mount(root, Note(parent=root, name="idea"))
        ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

        note.set_body("# Idea\n\nbuilds on prior work\n")
        assert "builds on prior work" in note.body()

        note.cite(ref)
        assert "smith2024" in note.read_index()  # citation is a markdown link
        assert Path(ref.resolve()) in {Path(p) for p in note.out_edges()}

    def test_cite_threads_role_into_typed_edge(self, tmp_path: Path) -> None:
        root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
        root.materialize()
        note = _mount(root, Note(parent=root, name="idea"))
        ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

        note.cite(ref, role="derived_from")

        typed = note.typed_out_edges()
        assert len(typed) == 1
        assert typed[0].role == "derived_from"
        assert Path(typed[0].target) == Path(str(ref.resolve()))


class TestReferenceConcept:
    """``ReferenceConcept`` typed meta + citation text."""

    def test_typed_meta_and_citation_round_trip(self, tmp_path: Path) -> None:
        root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
        root.materialize()
        ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

        ref.write_reference_meta(ReferenceMeta(title="T", doi="10.1/x", year=2024))
        got = ref.read_ref_meta()
        assert isinstance(got, ReferenceMeta)
        assert got.title == "T"
        assert got.doi == "10.1/x"
        assert got.year == 2024

        ref.set_citation("Smith et al. 2024")
        assert ref.citation() == "Smith et al. 2024"

    def test_write_ref_meta_is_deprecated_alias(self, tmp_path: Path) -> None:
        root = Folder(name="bundle", kind="bundle.concept", root_path=str(tmp_path))
        root.materialize()
        ref = _mount(root, ReferenceConcept(parent=root, name="smith2024"))

        with pytest.warns(DeprecationWarning, match="write_reference_meta"):
            ref.write_ref_meta(ReferenceMeta(title="Alias", year=2024))

        assert ref.read_ref_meta().title == "Alias"
