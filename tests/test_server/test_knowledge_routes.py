"""Tests for the knowledge routes (``/api/knowledge``) — OKF Concept browse."""

from __future__ import annotations

from molexp.workspace.concepts import Note, ReferenceConcept
from molexp.workspace.reference_meta import ReferenceMeta


def _seed_concepts(workspace) -> None:
    note = workspace.add_folder(Note(parent=workspace, name="cg-notes"))
    note.set_body("# CG notes\n\nNotes on coarse-grained zwitterions.")
    ref = workspace.add_folder(ReferenceConcept(parent=workspace, name="kremer1990"))
    ref.write_ref_meta(
        ReferenceMeta(
            title="Dynamics of entangled linear polymer melts",
            authors=("Kremer", "Grest"),
            year=1990,
            doi="10.1063/1.458541",
            venue="J. Chem. Phys.",
        )
    )
    ref.set_citation("Kremer & Grest, JCP 1990")


def test_list_knowledge_empty_workspace(client):
    resp = client.get("/api/knowledge")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"notes": [], "references": [], "total": 0}


def test_list_knowledge_returns_notes_and_references(client, workspace):
    _seed_concepts(workspace)

    resp = client.get("/api/knowledge")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["notes"]) == 1
    assert len(body["references"]) == 1
    note = body["notes"][0]
    assert note["name"] == "cg-notes"
    assert "coarse-grained" in note["excerpt"]
    ref = body["references"][0]
    assert ref["title"] == "Dynamics of entangled linear polymer melts"
    assert ref["authors"] == ["Kremer", "Grest"]
    assert ref["year"] == 1990


def test_get_note_returns_body(client, workspace):
    _seed_concepts(workspace)

    resp = client.get("/api/knowledge/note", params={"path": "cg-notes"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "cg-notes"
    assert "Notes on coarse-grained zwitterions" in body["body"]


def test_get_note_404_for_unknown(client):
    resp = client.get("/api/knowledge/note", params={"path": "nope"})
    assert resp.status_code == 404
