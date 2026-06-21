"""Tests for the per-scope notes + references library."""

from __future__ import annotations

import json

from molexp.workspace import NoteAsset, Reference
from molexp.workspace.assets import AssetScope


def test_add_note_registers_asset_and_writes_markdown(project):
    note = project.library.add_note(
        "Gold standard decision",
        "# Gold standard\n\nfp32 is the practical baseline.\n",
        summary="Why fp32 (not fp64) anchors ΔF",
        tags=["decision", "quantization"],
        refs=["fennol2024"],
    )

    assert isinstance(note, NoteAsset)
    assert note.kind == "note"
    assert note.path.as_posix() == "library/notes/gold-standard-decision.md"
    assert note.summary == "Why fp32 (not fp64) anchors ΔF"
    assert note.refs == ("fennol2024",)

    # File on disk under the project's library/.
    body = project.library.read_note(note)
    assert "practical baseline" in body

    # Registered in the authoritative manifest AND the derived catalog.
    assert [a.asset_id for a in project.library.list_notes()] == [note.asset_id]
    catalog_assets = project.assets.query(kind="note")
    assert [a.asset_id for a in catalog_assets] == [note.asset_id]


def test_add_note_is_idempotent_on_slug(project):
    first = project.library.add_note("My Note", "v1")
    second = project.library.add_note("My Note", "v2")

    assert first.asset_id == second.asset_id  # same path -> same asset id
    assert project.library.read_note(second) == "v2"
    assert len(project.library.list_notes()) == 1


def test_update_note_rewrites_body_preserving_metadata(project):
    note = project.library.add_note("Editable", "v1 body", summary="keeps", tags=["a"], refs=["r1"])
    updated = project.library.update_note(note.asset_id, "# v2\n\nnew body")

    assert updated.asset_id == note.asset_id
    assert updated.summary == "keeps"
    assert tuple(updated.tags.keys()) == ("a",)
    assert updated.refs == ("r1",)
    assert project.library.read_note(updated) == "# v2\n\nnew body"
    assert updated.content_hash != note.content_hash


def test_update_discovered_note_edits_file_in_place(project):
    readme = project.project_dir / "README.md"
    readme.write_text("# Old\n")
    [note] = project.library.discover_notes()
    project.library.update_note(note.asset_id, "# New\n")
    assert readme.read_text() == "# New\n"  # original file edited, not a copy


def test_update_unknown_note_raises(project):
    import pytest

    with pytest.raises(KeyError):
        project.library.update_note("no-such-id", "x")


def test_notes_survive_catalog_rebuild(project, workspace):
    note = project.library.add_note("Persisted", "body")
    workspace.catalog.rebuild()
    rebuilt = workspace.catalog.get(note.asset_id)
    assert rebuilt is not None
    assert rebuilt.kind == "note"


def test_reference_store_roundtrip_and_idempotent(project):
    ref = Reference(
        key="so3krates2026",
        title="8-bit QAT of an SO(3)-equivariant transformer",
        authors=("Frank",),
        year=2026,
        arxiv="2601.02213",
        tags=("quantization", "mlip"),
        note="Only direct precedent for MLIP quantization.",
    )
    project.library.add_reference(ref)
    project.library.add_reference(ref.model_copy(update={"year": 2025}))  # overwrite

    refs = project.library.list_references()
    assert len(refs) == 1
    assert refs[0].year == 2025
    assert refs[0].best_url == "https://arxiv.org/abs/2601.02213"


def test_build_index_writes_json_and_markdown(project):
    project.library.add_note(
        "Literature review", "...", summary="positioning vs prior art", refs=["so3krates2026"]
    )
    project.library.add_reference(
        Reference(key="so3krates2026", title="So3krates QAT", arxiv="2601.02213")
    )
    index = project.library.build_index()

    assert len(index.notes) == 1
    assert len(index.references) == 1

    idx_json = project.project_dir / "library" / "index.json"
    idx_md = project.project_dir / "library" / "INDEX.md"
    assert idx_json.exists()
    assert idx_md.exists()

    data = json.loads(idx_json.read_text())
    assert data["scope"] == "project/test-project"
    assert data["notes"][0]["title"] == "Literature review"

    md = idx_md.read_text()
    assert "Literature review" in md
    assert "so3krates2026" in md


def test_discovers_loose_markdown_in_scope_dir(project):
    # A README dropped directly in the project dir (not via add_note).
    (project.project_dir / "README.md").write_text("# pinet-quant\n\nProject overview.\n")
    discovered = project.library.discover_notes()

    assert len(discovered) == 1
    note = discovered[0]
    assert note.title == "pinet-quant"  # parsed from the first H1
    assert note.path.as_posix() == "README.md"
    assert note.tags.get("discovered") == "1"
    # Registered like any asset → catalog-visible + agent-discoverable.
    assert project.assets.query(kind="note")[0].asset_id == note.asset_id


def test_discovery_prunes_deleted_files(project):
    readme = project.project_dir / "README.md"
    readme.write_text("# Doc\n")
    project.library.discover_notes()
    assert len(project.library.list_notes()) == 1

    readme.unlink()
    project.library.discover_notes()
    assert project.library.list_notes() == []


def test_build_index_runs_discovery(project):
    (project.project_dir / "NOTES.md").write_text("# Field notes\n")
    index = project.library.build_index()
    assert any(n.title == "Field notes" for n in index.notes)


def test_library_available_on_every_scope(workspace, project, experiment, run):
    for scope_obj, expected_kind in (
        (workspace, "workspace"),
        (project, "project"),
        (experiment, "experiment"),
        (run, "run"),
    ):
        note = scope_obj.library.add_note("Scoped", "body")
        assert isinstance(note.scope, AssetScope)
        assert note.scope.kind == expected_kind
        assert scope_obj.library.list_notes()[0].asset_id == note.asset_id
