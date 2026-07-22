"""Document-embed target resolution + entity summaries (``workspace.doc_embed``).

Covers the workspace-layer enrichment surface for OKF ``Note`` documents:

- ``Bundle.embed(note, target, *, role=None)`` — write ONE typed provenance
  edge from a ``Note`` to a live entity (``Run`` / ``Asset`` / ``Experiment`` /
  ``ReferenceConcept``) with a per-kind default role from the frozen
  ``EdgeRole`` vocabulary; the asset payload is pointed at, never copied.
- ``summarize_entity`` / ``EntitySummary`` — a pure read projection of an
  entity's ``id`` / ``kind`` / ``title`` (the UI card source).
- ``asset_record_dir`` — anchor an ``Asset`` to its in-tree record dir, raising
  (never falling back) when the root is missing or the dir does not exist.
- ``Note`` ``tags`` / ``status`` meta helpers backed by ``NoteMeta`` (this file
  is their sole owner — the concepts suite covers body/cite/reference meta).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from molexp.workspace import (
    Bundle,
    EntitySummary,
    NoteMeta,
    ReferenceConcept,
    ReferenceMeta,
    Workspace,
    summarize_entity,
)
from molexp.workspace.doc_embed import asset_record_dir

_VALID_ROLES = {"derived_from", "cites", "supersedes", "records", "references"}


def _norm(path: object) -> str:
    return os.path.normpath(str(path))


@pytest.fixture
def scene(tmp_path: Path) -> SimpleNamespace:
    """A workspace with a project / experiment / run, a data asset, a reference, a note."""
    ws = Workspace(root=tmp_path / "lab")
    ws.materialize()
    proj = ws.add_project("p")
    exp = proj.add_experiment("e")
    run = exp.add_run(id="r")

    # A user DataAsset in the experiment scope: record dir at
    # ``<exp_dir>/assets/<asset_id>/`` (the same location assets.scan reads).
    src = tmp_path / "input.txt"
    src.write_text("payload-bytes")
    asset = exp.data_assets.import_asset("mydata", src)

    # A ReferenceConcept mounted at the bundle root; its title lives in
    # ReferenceMeta (no index.md H1).
    ref = cast("ReferenceConcept", ws.add_folder(ReferenceConcept(parent=ws, name="smith2024")))
    ref.write_reference_meta(ReferenceMeta(title="Smith 2024", year=2024))

    b = Bundle(ws.resolve())
    doc = b.create_note("My Doc", body="# My Doc\n\nbody")

    return SimpleNamespace(
        ws=ws, proj=proj, exp=exp, run=run, asset=asset, ref=ref, b=b, doc=doc, root=ws.resolve()
    )


class TestBundleEmbed:
    """``Bundle.embed`` — one typed provenance edge from a Note to a live entity."""

    def test_embed_run_writes_a_relative_records_edge(self, scene: SimpleNamespace) -> None:
        scene.b.embed(scene.doc, scene.run)

        edges = scene.doc.typed_out_edges()
        matched = [e for e in edges if _norm(e.target) == _norm(scene.run.resolve())]
        assert len(matched) == 1
        assert matched[0].role == "records"

        # The raw index.md line is a *relative* markdown link (never absolute).
        raw = scene.doc.read_index()
        expected_rel = os.path.relpath(str(scene.run.resolve()), str(scene.doc.resolve()))
        expected_posix = Path(expected_rel).as_posix()
        assert f"]({expected_posix})" in raw
        assert not expected_posix.startswith("/")

    def test_embed_default_role_per_target_kind(self, scene: SimpleNamespace) -> None:
        scene.b.embed(scene.doc, scene.run)
        scene.b.embed(scene.doc, scene.asset)
        scene.b.embed(scene.doc, scene.exp)
        scene.b.embed(scene.doc, scene.ref)

        edges = scene.doc.typed_out_edges()
        by_target = {_norm(e.target): e.role for e in edges}

        record_dir = asset_record_dir(scene.asset, scene.root)
        assert by_target[_norm(scene.run.resolve())] == "records"
        assert by_target[_norm(scene.exp.resolve())] == "records"
        assert by_target[_norm(scene.ref.resolve())] == "cites"
        assert by_target[_norm(record_dir)] == "references"

        # Every recovered role is a member of the frozen EdgeRole vocabulary,
        # and four in-tree relative links were recovered (one per target kind).
        assert all(e.role in _VALID_ROLES for e in edges)
        assert len(edges) == 4

    def test_explicit_role_overrides_the_per_kind_default(self, scene: SimpleNamespace) -> None:
        scene.b.embed(scene.doc, scene.run, role="derived_from")

        edges = scene.doc.typed_out_edges()
        matched = [e for e in edges if _norm(e.target) == _norm(scene.run.resolve())]
        assert len(matched) == 1
        assert matched[0].role == "derived_from"

    def test_embed_asset_points_at_record_dir_without_copying(self, scene: SimpleNamespace) -> None:
        record_dir = asset_record_dir(scene.asset, scene.root)
        before = sorted(p.name for p in Path(record_dir).iterdir())

        scene.b.embed(scene.doc, scene.asset)

        after = sorted(p.name for p in Path(record_dir).iterdir())
        assert before == after  # the asset record dir gained no doc copy


class TestSummarizeEntity:
    """``summarize_entity`` — pure id / kind / title projection per target kind."""

    def test_summary_per_target_kind(self, scene: SimpleNamespace) -> None:
        s_run = summarize_entity(scene.run)
        assert isinstance(s_run, EntitySummary)
        assert (s_run.id, s_run.kind, s_run.title) == ("r", "workspace.run", "r")

        s_exp = summarize_entity(scene.exp)
        assert (s_exp.id, s_exp.kind, s_exp.title) == ("e", "workspace.experiment", "e")

        s_ref = summarize_entity(scene.ref)
        assert (s_ref.id, s_ref.kind) == ("smith2024", "reference.reference")
        assert s_ref.title == "Smith 2024"  # ReferenceMeta.title preferred over the name

        s_asset = summarize_entity(scene.asset, root=scene.root)
        assert (s_asset.id, s_asset.kind, s_asset.title) == (
            scene.asset.asset_id,
            "data",
            "mydata",
        )

        # Bundle.entity_summary is the convenience wrapper passing the bundle root.
        assert scene.b.entity_summary(scene.asset) == s_asset

    def test_folder_title_prefers_index_h1(self, scene: SimpleNamespace) -> None:
        scene.run.write_index("# Nice Run Title\n\nnotes")
        assert summarize_entity(scene.run).title == "Nice Run Title"

    def test_asset_target_without_root_raises(self, scene: SimpleNamespace) -> None:
        with pytest.raises(ValueError):
            summarize_entity(scene.asset)  # Asset target needs root anchoring


class TestAssetRecordDir:
    """``asset_record_dir`` — anchor an Asset, never fall back on a missing dir."""

    def test_missing_record_dir_raises(self, scene: SimpleNamespace) -> None:
        ghost = scene.asset.model_copy(update={"asset_id": "does-not-exist-0000"})
        with pytest.raises(FileNotFoundError):
            asset_record_dir(ghost, scene.root)


class TestNoteMeta:
    """``Note`` tags / status helpers backed by ``NoteMeta`` (sole owner here)."""

    def test_tags_and_status_round_trip_through_meta_yaml(self, scene: SimpleNamespace) -> None:
        scene.doc.set_tags(["a"])
        scene.doc.set_status("archived")

        meta = scene.doc.read_note_meta()
        assert isinstance(meta, NoteMeta)
        assert meta.tags == ["a"]
        assert meta.status == "archived"

        # The convenience accessors agree with the typed meta.
        assert scene.doc.tags() == ["a"]
        assert scene.doc.status() == "archived"

    def test_partial_update_preserves_the_sibling_field(self, scene: SimpleNamespace) -> None:
        scene.doc.set_status("archived")
        scene.doc.set_tags(["x", "y"])  # must not clobber status
        assert scene.doc.status() == "archived"
        assert scene.doc.tags() == ["x", "y"]

        scene.doc.set_status("active")  # must not clobber tags
        assert scene.doc.tags() == ["x", "y"]

    def test_bare_marker_defaults_to_untagged_active(self, scene: SimpleNamespace) -> None:
        note = scene.b.create_note("Bare")
        note.write_meta()  # write exactly {type, id} — the legacy bare marker

        meta = note.read_note_meta()
        assert meta.tags == []
        assert meta.status == "active"
