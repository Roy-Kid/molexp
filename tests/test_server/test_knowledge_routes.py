"""Tests for the knowledge routes (``/api/knowledge``) — OKF Concept browse + docs."""

from __future__ import annotations

import pytest

from molexp.workspace import Bundle
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


# ── read-only endpoints (pre-existing) ───────────────────────────────────────


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


# ── ac-001 — POST /knowledge/doc → 201 + NoteSummary ─────────────────────────


def test_create_doc_returns_201_with_summary(client, workspace):
    resp = client.post(
        "/api/knowledge/doc",
        json={"name": "new-doc", "body": "hello world"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "new-doc"
    assert body["relPath"] == "new-doc"
    assert "hello world" in body["excerpt"]

    # It materialized on disk and is now browsable.
    got = client.get("/api/knowledge/note", params={"path": "new-doc"})
    assert got.status_code == 200
    assert "hello world" in got.json()["body"]


def test_create_doc_under_parent(client, workspace):
    client.post("/api/knowledge/doc", json={"name": "parent-doc"})
    resp = client.post(
        "/api/knowledge/doc",
        json={"name": "child-doc", "parentPath": "parent-doc", "body": "nested"},
    )
    assert resp.status_code == 201
    assert resp.json()["relPath"] == "parent-doc/child-doc"


# ── ac-002 — PUT /knowledge/doc?path= persists body to index.md ──────────────


def test_edit_doc_persists_body_to_index_md(client, workspace):
    _seed_concepts(workspace)

    resp = client.put(
        "/api/knowledge/doc",
        params={"path": "cg-notes"},
        json={"body": "# Rewritten\n\nfresh body text"},
    )
    assert resp.status_code == 200
    detail = resp.json()
    assert "fresh body text" in detail["body"]

    # Re-read via GET confirms persistence.
    got = client.get("/api/knowledge/note", params={"path": "cg-notes"})
    assert "fresh body text" in got.json()["body"]

    # And the bytes actually landed in the note's index.md on disk.
    index_md = workspace.root / "cg-notes" / "index.md"
    assert "fresh body text" in index_md.read_text()


# ── ac-003 — PATCH /knowledge/doc?path= rename / move ────────────────────────


def test_move_doc_rename_new_path_resolves_old_404(client, workspace):
    _seed_concepts(workspace)

    resp = client.patch(
        "/api/knowledge/doc",
        params={"path": "cg-notes"},
        json={"name": "renamed-notes"},
    )
    assert resp.status_code == 200
    assert resp.json()["relPath"] == "renamed-notes"

    assert client.get("/api/knowledge/note", params={"path": "renamed-notes"}).status_code == 200
    assert client.get("/api/knowledge/note", params={"path": "cg-notes"}).status_code == 404


def test_move_doc_reparent(client, workspace):
    client.post("/api/knowledge/doc", json={"name": "home"})
    client.post("/api/knowledge/doc", json={"name": "loose-doc"})

    resp = client.patch(
        "/api/knowledge/doc",
        params={"path": "loose-doc"},
        json={"parentPath": "home"},
    )
    assert resp.status_code == 200
    assert resp.json()["relPath"] == "home/loose-doc"
    assert client.get("/api/knowledge/note", params={"path": "home/loose-doc"}).status_code == 200
    assert client.get("/api/knowledge/note", params={"path": "loose-doc"}).status_code == 404


# ── ac-004 — DELETE /knowledge/doc?path= → MessageResponse, then GET 404 ─────


def test_delete_doc_then_get_404(client, workspace):
    _seed_concepts(workspace)

    resp = client.delete("/api/knowledge/doc", params={"path": "cg-notes"})
    assert resp.status_code == 200
    assert "message" in resp.json()

    assert client.get("/api/knowledge/note", params={"path": "cg-notes"}).status_code == 404


# ── ac-005 — GET /knowledge/backlinks?path= ──────────────────────────────────


def test_backlinks_returns_linking_concepts(client, workspace):
    note_a = workspace.add_folder(Note(parent=workspace, name="target-a"))
    note_a.set_body("# A")
    note_b = workspace.add_folder(Note(parent=workspace, name="source-b"))
    note_b.set_body("# B")
    note_b.cite(note_a)

    resp = client.get("/api/knowledge/backlinks", params={"path": "target-a"})
    assert resp.status_code == 200
    backlinks = resp.json()["backlinks"]
    names = {b["name"] for b in backlinks}
    assert "source-b" in names


# ── ac-006 — GET /knowledge/doc/export?path= → text/markdown ─────────────────


def test_export_doc_returns_markdown(client, workspace):
    _seed_concepts(workspace)

    resp = client.get("/api/knowledge/doc/export", params={"path": "cg-notes"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "content-disposition" in {k.lower() for k in resp.headers}
    assert "coarse-grained zwitterions" in resp.text


# ── ac-007 — write-gate: remote served workspace → 405 ───────────────────────


@pytest.fixture
def _remote_served():
    """Make the active served workspace remote (read-only), then reset."""
    from molexp.server.dependencies import (
        ServedWorkspace,
        set_active_workspace_descriptor,
        set_served_workspaces,
    )

    set_served_workspaces(
        [
            ServedWorkspace(
                key="remote-ws", label="me@hpc:/runs", is_remote=True, target_name="remote-ws"
            )
        ]
    )
    set_active_workspace_descriptor("remote-ws")
    try:
        yield
    finally:
        set_active_workspace_descriptor(None)
        set_served_workspaces([])


def test_writable_gate_blocks_remote(client, workspace, _remote_served):
    assert client.post("/api/knowledge/doc", json={"name": "x"}).status_code == 405
    assert (
        client.put(
            "/api/knowledge/doc", params={"path": "cg-notes"}, json={"body": "y"}
        ).status_code
        == 405
    )
    assert (
        client.patch(
            "/api/knowledge/doc", params={"path": "cg-notes"}, json={"name": "z"}
        ).status_code
        == 405
    )
    assert client.delete("/api/knowledge/doc", params={"path": "cg-notes"}).status_code == 405


def test_writable_gate_local_stays_writable(client, workspace):
    # No remote served workspace active: a mutating call is not 405.
    resp = client.post("/api/knowledge/doc", json={"name": "local-doc"})
    assert resp.status_code != 405
    assert resp.status_code == 201


# ── ac-008 — ConceptNotFoundError → 404 on every path-addressed handler ──────


def test_unknown_path_maps_to_404(client, workspace):
    unknown = {"path": "does-not-exist"}
    assert client.put("/api/knowledge/doc", params=unknown, json={"body": "x"}).status_code == 404
    assert client.patch("/api/knowledge/doc", params=unknown, json={"name": "y"}).status_code == 404
    assert client.delete("/api/knowledge/doc", params=unknown).status_code == 404
    assert client.get("/api/knowledge/backlinks", params=unknown).status_code == 404
    assert client.get("/api/knowledge/doc/export", params=unknown).status_code == 404


def test_non_note_concept_maps_to_404(client, workspace):
    _seed_concepts(workspace)  # kremer1990 is a ReferenceConcept, not a Note
    assert (
        client.put(
            "/api/knowledge/doc", params={"path": "kremer1990"}, json={"body": "x"}
        ).status_code
        == 404
    )


# ── ac-009 — Python==UI parity: thin delegation to Bundle verbs ──────────────


def test_parity_handlers_delegate_to_bundle_verbs(client, workspace, monkeypatch):
    _seed_concepts(workspace)
    seen: dict[str, bool] = {}

    def _spy(name, cls, attr):
        orig = getattr(cls, attr)

        def wrapper(self, *args: object, **kwargs: object):
            seen[name] = True
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(cls, attr, wrapper)

    _spy("create_note", Bundle, "create_note")
    _spy("set_body", Note, "set_body")
    _spy("rename_note", Bundle, "rename_note")
    _spy("move_note", Bundle, "move_note")
    _spy("delete_note", Bundle, "delete_note")
    _spy("backlinks", Bundle, "backlinks")
    _spy("export_markdown", Bundle, "export_markdown")

    client.post("/api/knowledge/doc", json={"name": "parity-doc", "body": "b"})
    assert seen.get("create_note")

    client.put("/api/knowledge/doc", params={"path": "cg-notes"}, json={"body": "edited"})
    assert seen.get("set_body")

    client.patch("/api/knowledge/doc", params={"path": "cg-notes"}, json={"name": "cg-renamed"})
    assert seen.get("rename_note")

    client.get("/api/knowledge/backlinks", params={"path": "cg-renamed"})
    assert seen.get("backlinks")

    client.get("/api/knowledge/doc/export", params={"path": "cg-renamed"})
    assert seen.get("export_markdown")

    client.delete("/api/knowledge/doc", params={"path": "cg-renamed"})
    assert seen.get("delete_note")


def test_parity_move_note_delegation(client, workspace, monkeypatch):
    client.post("/api/knowledge/doc", json={"name": "home"})
    client.post("/api/knowledge/doc", json={"name": "mover"})

    seen: dict[str, bool] = {}
    orig = Bundle.move_note

    def wrapper(self, *args: object, **kwargs: object):
        seen["move_note"] = True
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Bundle, "move_note", wrapper)

    client.patch("/api/knowledge/doc", params={"path": "mover"}, json={"parentPath": "home"})
    assert seen.get("move_note")


# ── ac-010 — OpenAPI surface carries the new doc paths ───────────────────────


def test_openapi_contains_doc_paths():
    from molexp.server.app import create_app

    schema = create_app().openapi()
    paths = schema["paths"]
    assert "/api/knowledge/doc" in paths
    assert "/api/knowledge/backlinks" in paths
    assert "/api/knowledge/doc/export" in paths
    # The three mutating verbs live on /api/knowledge/doc.
    for verb in ("post", "put", "patch", "delete"):
        assert verb in paths["/api/knowledge/doc"]
