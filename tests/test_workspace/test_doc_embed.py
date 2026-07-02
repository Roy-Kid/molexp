"""RED tests for the document-embed verb + entity summary (knowledge-docs-05).

Covers the workspace-layer enrichment surface for OKF Note documents:

- ``Bundle.embed(src_concept, target, *, role=None)`` — write ONE typed
  provenance edge from a ``Note`` to a live entity (``Run`` / ``Asset`` /
  ``Experiment`` / ``ReferenceConcept``) through the single ``append_link``
  edge-writer, with a per-kind default role from the frozen ``EdgeRole``
  vocabulary.
- ``summarize_entity`` / ``EntitySummary`` — a pure read projection of an
  entity's ``id`` / ``kind`` / ``title`` (the future UI card source).
- ``Note`` ``tags`` / ``status`` meta helpers backed by ``NoteMeta``.

None of these symbols exist yet, so this module fails at import (collection)
time until the production surface lands — the desired RED state.

Conventions mirror the sibling ``test_bundle.py`` / ``test_bundle_docs.py``:
``from __future__ import annotations``, build bundles + entities on
``tmp_path``, and assert against resolved paths rather than raw markdown.

Acceptance map:
    ac-001  embed to a Run writes a ``records`` typed edge (relative link)
    ac-002  embed supports Run / Asset / Experiment / ReferenceConcept
    ac-003  embed writes edges only via ``folder.append_link`` (byte-equal)
    ac-004  explicit ``role=`` overrides the per-kind default
    ac-005  summarize_entity returns id / kind / title per target kind
    ac-006  Note tags / status round-trip through ``meta.yaml``
    ac-007  bare ``note.note`` marker defaults to untagged / active
    ac-008  new public API exported on ``molexp.workspace.__all__``
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import molexp.workspace as mw
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
from molexp.workspace.folder import append_link

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


# ── ac-001: embed to a Run writes a records typed edge ───────────────────────


def test_embed_run_writes_records_edge(scene: SimpleNamespace) -> None:
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


# ── ac-002: embed supports Run / Asset / Experiment / ReferenceConcept ───────


def test_embed_each_target_kind_recovered(scene: SimpleNamespace) -> None:
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

    # Every recovered role is a member of the frozen EdgeRole vocabulary.
    assert all(e.role in _VALID_ROLES for e in edges)
    # Four in-tree relative links were recovered (one per target kind).
    assert len(edges) == 4


# ── ac-003: embed writes edges only via folder.append_link (byte-equal) ──────


def test_embed_routes_through_append_link(scene: SimpleNamespace) -> None:
    base = scene.doc.read_index()

    scene.b.embed(scene.doc, scene.run)
    embed_out = scene.doc.read_index()

    # Reset to the same base, then reproduce the line by hand via the single
    # edge-writer with the same target + role.
    scene.doc.write_index(base)
    append_link(scene.doc, scene.run, role="records")
    manual_out = scene.doc.read_index()

    assert embed_out == manual_out


# ── ac-004: explicit role= overrides the per-kind default ────────────────────


def test_embed_explicit_role_overrides_default(scene: SimpleNamespace) -> None:
    scene.b.embed(scene.doc, scene.run, role="derived_from")

    edges = scene.doc.typed_out_edges()
    matched = [e for e in edges if _norm(e.target) == _norm(scene.run.resolve())]
    assert len(matched) == 1
    assert matched[0].role == "derived_from"


# ── ac-005: summarize_entity returns id / kind / title per target kind ───────


def test_summarize_entity_per_kind(scene: SimpleNamespace) -> None:
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


def test_summarize_run_uses_index_h1_title_when_present(scene: SimpleNamespace) -> None:
    # A Run with an index.md H1 uses that as the title (over its bare name).
    scene.run.write_index("# Nice Run Title\n\nnotes")
    s_run = summarize_entity(scene.run)
    assert s_run.title == "Nice Run Title"


# ── ac-006: Note tags / status round-trip through meta.yaml ───────────────────


def test_note_tags_status_round_trip(scene: SimpleNamespace) -> None:
    scene.doc.set_tags(["a"])
    scene.doc.set_status("archived")

    meta = scene.doc.read_note_meta()
    assert isinstance(meta, NoteMeta)
    assert meta.tags == ["a"]
    assert meta.status == "archived"

    # The convenience accessors agree with the typed meta.
    assert scene.doc.tags() == ["a"]
    assert scene.doc.status() == "archived"


def test_note_set_tags_preserves_status_and_vice_versa(scene: SimpleNamespace) -> None:
    scene.doc.set_status("archived")
    scene.doc.set_tags(["x", "y"])  # must not clobber status
    assert scene.doc.status() == "archived"
    assert scene.doc.tags() == ["x", "y"]

    scene.doc.set_status("active")  # must not clobber tags
    assert scene.doc.tags() == ["x", "y"]


# ── ac-007: bare note.note marker defaults to untagged / active ──────────────


def test_note_bare_marker_defaults_untagged_active(scene: SimpleNamespace) -> None:
    note = scene.b.create_note("Bare")
    note.write_meta()  # write exactly {type, id} — the legacy bare marker

    meta = note.read_note_meta()
    assert meta.tags == []
    assert meta.status == "active"


# ── ac-008: new public API exported on molexp.workspace.__all__ ──────────────


def test_public_exports() -> None:
    for name in ("NoteMeta", "EntitySummary", "summarize_entity"):
        assert name in mw.__all__, f"{name} missing from molexp.workspace.__all__"
        assert hasattr(mw, name), f"{name} not importable from molexp.workspace"


# ── Edge cases (spec Testing strategy) ───────────────────────────────────────


def test_embed_asset_does_not_copy_payload(scene: SimpleNamespace) -> None:
    record_dir = asset_record_dir(scene.asset, scene.root)
    before = sorted(p.name for p in Path(record_dir).iterdir())

    scene.b.embed(scene.doc, scene.asset)

    after = sorted(p.name for p in Path(record_dir).iterdir())
    assert before == after  # the asset record dir gained no doc copy


def test_summarize_asset_without_root_raises(scene: SimpleNamespace) -> None:
    with pytest.raises(ValueError):
        summarize_entity(scene.asset)  # Asset target needs root anchoring


def test_asset_record_dir_missing_raises(scene: SimpleNamespace) -> None:
    # An Asset whose scope has no on-disk record dir must raise, never fall back.
    ghost = scene.asset.model_copy(update={"asset_id": "does-not-exist-0000"})
    with pytest.raises(FileNotFoundError):
        asset_record_dir(ghost, scene.root)
