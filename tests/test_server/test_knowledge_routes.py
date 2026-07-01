"""Tests for the knowledge routes (``/api/knowledge``) — OKF Concept browse + docs."""

from __future__ import annotations

import os

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


# ── knowledge-docs-06: embed routes, summary cards, tag/status ───────────────


def _seed_run(workspace, *, run_id: str = "r"):
    """Seed a project / experiment / run and return the Run folder."""
    proj = workspace.add_project("p")
    exp = proj.add_experiment("e")
    return exp.add_run(id=run_id)


def _seed_note(workspace, name: str = "doc", *, body: str = "# Doc"):
    """Seed a root-mounted Note document via the Bundle façade."""
    return Bundle(workspace.root).create_note(name, body=body)


# ── ac-001 — POST /knowledge/doc/embed writes a typed provenance edge ────────


def test_embed_writes_edge(client, workspace):
    run = _seed_run(workspace, run_id="r")
    _seed_note(workspace, "doc")

    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "run", "target": "r", "role": "records"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["srcPath"] == "doc"
    assert data["target"] == "r"
    assert data["role"] == "records"

    # The edge landed on the source note's typed out-edges, resolving to the run.
    fresh = Bundle(workspace.root).get("doc")
    edges = fresh.typed_out_edges()
    assert len(edges) == 1
    assert edges[0].role == "records"
    assert os.path.normpath(edges[0].target) == os.path.normpath(str(run.resolve()))


def test_embed_no_role_uses_per_kind_default_parity(client, workspace):
    """No wire role -> the SAME per-kind default a CLI ``Bundle.embed`` writes.

    Embedding a Run with no explicit role must yield ``records`` (Bundle.embed's
    per-kind default), not a route-local ``references`` default — otherwise the
    HTTP path and the Python/CLI path would write different edges for the same
    operation, violating the Python==UI invariant.
    """
    _seed_run(workspace, run_id="r")
    _seed_note(workspace, "doc")

    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "run", "target": "r"},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "records"
    assert Bundle(workspace.root).get("doc").typed_out_edges()[0].role == "records"


# ── ac-002 — embed delegates to Bundle.embed (Python==UI parity) ─────────────


def test_embed_delegates_to_bundle(client, workspace, monkeypatch):
    from molexp.workspace.run import Run

    _seed_run(workspace, run_id="r")
    _seed_note(workspace, "doc")

    calls: list[tuple[object, object, object]] = []
    orig = Bundle.embed

    def spy(self, src, target, *, role=None):
        calls.append((src, target, role))
        return orig(self, src, target, role=role)

    monkeypatch.setattr(Bundle, "embed", spy)

    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "run", "target": "r", "role": "records"},
    )
    assert resp.status_code == 200
    assert len(calls) == 1
    src, target, role = calls[0]
    assert isinstance(src, Note)
    assert isinstance(target, Run)
    assert target.name == "r"
    assert role == "records"


# ── ac-003 — embed route is writable-gated (read-only workspace rejected) ────


def test_embed_writable_gate(client, workspace, _remote_served):
    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "run", "target": "r"},
    )
    assert resp.status_code == 405


def test_embed_writable_gate_local_not_405(client, workspace):
    # A local workspace is not rejected by the gate (404 for the missing note,
    # never 405).
    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "nope"},
        json={"target_kind": "run", "target": "r"},
    )
    assert resp.status_code != 405


# ── ac-004 — unknown source note or missing target entity returns 404 ────────


def test_embed_404_unknown_source_note(client, workspace):
    _seed_run(workspace, run_id="r")
    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "no-such-note"},
        json={"target_kind": "run", "target": "r"},
    )
    assert resp.status_code == 404


def test_embed_404_non_note_source(client, workspace):
    _seed_concepts(workspace)  # kremer1990 is a ReferenceConcept, not a Note
    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "kremer1990"},
        json={"target_kind": "reference", "target": "kremer1990"},
    )
    assert resp.status_code == 404


def test_embed_404_missing_target_entity(client, workspace):
    _seed_note(workspace, "doc")
    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "run", "target": "ghost"},
    )
    assert resp.status_code == 404


def test_embed_404_missing_asset_target(client, workspace):
    _seed_note(workspace, "doc")
    resp = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "asset", "target": "no-such-asset"},
    )
    assert resp.status_code == 404


# ── ac-005 — GET /knowledge/note resolves embedded-entity summary cards ──────


def test_note_summary_cards(client, workspace):
    run = _seed_run(workspace, run_id="r")
    _seed_note(workspace, "doc")

    embed = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "run", "target": "r", "role": "records"},
    )
    assert embed.status_code == 200

    got = client.get("/api/knowledge/note", params={"path": "doc"})
    assert got.status_code == 200
    cards = got.json()["cards"]
    assert len(cards) == 1
    card = cards[0]
    assert card["kind"] == "workspace.run"
    assert card["id"] == "r"
    assert card["title"]  # non-empty resolved title
    assert card["status"] is None  # a Run has no note status
    assert os.path.normpath(str(run.resolve())).endswith(os.path.normpath(card["relPath"]))


def test_note_summary_cards_asset_target(client, workspace, tmp_path):
    proj = workspace.add_project("p")
    exp = proj.add_experiment("e")
    src = tmp_path / "input.txt"
    src.write_text("payload")
    asset = exp.data_assets.import_asset("mydata", str(src))
    _seed_note(workspace, "doc")

    embed = client.post(
        "/api/knowledge/doc/embed",
        params={"path": "doc"},
        json={"target_kind": "asset", "target": asset.asset_id},
    )
    assert embed.status_code == 200

    cards = client.get("/api/knowledge/note", params={"path": "doc"}).json()["cards"]
    assert len(cards) == 1
    assert cards[0]["id"] == asset.asset_id


# ── ac-006 — GET /knowledge exposes tags and status on note summaries ────────


def test_knowledge_list_tags_status(client, workspace):
    note = _seed_note(workspace, "tagged", body="# Tagged")
    note.set_tags(["physics", "md"])
    note.set_status("draft")

    resp = client.get("/api/knowledge")
    assert resp.status_code == 200
    summaries = {n["name"]: n for n in resp.json()["notes"]}
    assert summaries["tagged"]["tags"] == ["physics", "md"]
    assert summaries["tagged"]["status"] == "draft"


# ── ac-007 — GET /knowledge tag/status query params narrow the list ──────────


def test_knowledge_list_filter(client, workspace):
    alpha = _seed_note(workspace, "alpha", body="# A")
    alpha.set_tags(["physics"])
    alpha.set_status("draft")
    beta = _seed_note(workspace, "beta", body="# B")
    beta.set_tags(["chem"])
    beta.set_status("active")

    def _names(**params: str):
        return {n["name"] for n in client.get("/api/knowledge", params=params).json()["notes"]}

    assert _names(tag="physics") == {"alpha"}
    assert _names(status="active") == {"beta"}
    # AND semantics when combined.
    assert _names(tag="physics", status="draft") == {"alpha"}
    assert _names(tag="physics", status="active") == set()


# ── ac-008 — OpenAPI surface carries the new embed path ──────────────────────


def test_openapi_contains_embed_path():
    from molexp.server.app import create_app

    schema = create_app().openapi()
    paths = schema["paths"]
    assert "/api/knowledge/doc/embed" in paths
    assert "post" in paths["/api/knowledge/doc/embed"]


# ── knowledge-docs-08: writable Note tags/status via PATCH /doc/meta ─────────


# ── ac-001 — PATCH /knowledge/doc/meta?path= with tags persists ──────────────


def test_patch_doc_meta_tags_persists(client, workspace):
    _seed_note(workspace, "doc", body="# Doc")

    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "doc"},
        json={"tags": ["physics", "md"]},
    )
    assert resp.status_code == 200
    assert resp.json()["tags"] == ["physics", "md"]

    # Re-read through the workspace API confirms it landed in meta.yaml.
    assert Bundle(workspace.root).get("doc").tags() == ["physics", "md"]


# ── ac-002 — PATCH /knowledge/doc/meta?path= with status persists ────────────


def test_patch_doc_meta_status_persists(client, workspace):
    _seed_note(workspace, "doc", body="# Doc")

    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "doc"},
        json={"status": "draft"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "draft"

    assert Bundle(workspace.root).get("doc").status() == "draft"


# ── ac-003 — partial PATCH preserves the untouched sibling field ─────────────


def test_patch_doc_meta_partial_preserves_sibling(client, workspace):
    note = _seed_note(workspace, "doc", body="# Doc")
    note.set_tags(["physics"])
    note.set_status("draft")

    # PATCH only status -> tags preserved.
    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "doc"},
        json={"status": "archived"},
    )
    assert resp.status_code == 200
    fresh = Bundle(workspace.root).get("doc")
    assert fresh.status() == "archived"
    assert fresh.tags() == ["physics"]

    # PATCH only tags -> status preserved.
    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "doc"},
        json={"tags": ["physics", "chem"]},
    )
    assert resp.status_code == 200
    fresh = Bundle(workspace.root).get("doc")
    assert fresh.tags() == ["physics", "chem"]
    assert fresh.status() == "archived"


# ── ac-004 — write-gate: remote served workspace → 405 ───────────────────────


def test_patch_doc_meta_writable_gate(client, workspace, _remote_served):
    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "doc"},
        json={"status": "draft"},
    )
    assert resp.status_code == 405


def test_patch_doc_meta_local_not_405(client, workspace):
    # A local workspace is not rejected by the gate (404 for the missing note).
    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "nope"},
        json={"status": "draft"},
    )
    assert resp.status_code != 405


# ── ac-005 — unknown path / non-note concept → 404 ───────────────────────────


def test_patch_doc_meta_404_unknown_path(client, workspace):
    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "does-not-exist"},
        json={"status": "draft"},
    )
    assert resp.status_code == 404


def test_patch_doc_meta_404_non_note(client, workspace):
    _seed_concepts(workspace)  # kremer1990 is a ReferenceConcept, not a Note
    resp = client.patch(
        "/api/knowledge/doc/meta",
        params={"path": "kremer1990"},
        json={"tags": ["x"]},
    )
    assert resp.status_code == 404


# ── ac-006 — parity: delegates to Note.set_tags / Note.set_status ────────────


def test_patch_doc_meta_delegates_to_note_verbs(client, workspace, monkeypatch):
    _seed_note(workspace, "doc", body="# Doc")

    seen: dict[str, object] = {}
    orig_tags = Note.set_tags
    orig_status = Note.set_status

    def tags_spy(self, tags):
        seen["set_tags"] = list(tags)
        return orig_tags(self, tags)

    def status_spy(self, status):
        seen["set_status"] = status
        return orig_status(self, status)

    monkeypatch.setattr(Note, "set_tags", tags_spy)
    monkeypatch.setattr(Note, "set_status", status_spy)

    # tags-only PATCH -> only set_tags fires.
    client.patch("/api/knowledge/doc/meta", params={"path": "doc"}, json={"tags": ["a"]})
    assert seen == {"set_tags": ["a"]}

    seen.clear()
    # status-only PATCH -> only set_status fires.
    client.patch("/api/knowledge/doc/meta", params={"path": "doc"}, json={"status": "draft"})
    assert seen == {"set_status": "draft"}


# ── ac-007 — OpenAPI surface carries PATCH /api/knowledge/doc/meta ────────────


def test_openapi_contains_doc_meta_patch():
    from molexp.server.app import create_app

    schema = create_app().openapi()
    paths = schema["paths"]
    assert "/api/knowledge/doc/meta" in paths
    assert "patch" in paths["/api/knowledge/doc/meta"]
