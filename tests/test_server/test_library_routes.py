"""Library routes — notes + references over HTTP."""

from __future__ import annotations


def test_get_empty_library(client, project):
    resp = client.get("/api/library", params={"project_id": project.id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == f"project/{project.id}"
    assert body["notes"] == []
    assert body["references"] == []


def test_add_note_then_index_and_content(client, project):
    resp = client.post(
        "/api/library/notes",
        params={"project_id": project.id},
        json={
            "title": "Gold standard decision",
            "content": "# Gold standard\n\nfp32 is the practical baseline.\n",
            "summary": "Why fp32 anchors ΔF",
            "tags": ["decision"],
            "refs": ["fennol2024"],
        },
    )
    assert resp.status_code == 201
    note = resp.json()
    assert note["path"] == "library/notes/gold-standard-decision.md"

    index = client.get("/api/library", params={"project_id": project.id}).json()
    assert index["notes"][0]["title"] == "Gold standard decision"

    # Note body is served by the asset content endpoint and preview-able.
    content = client.get(f"/api/assets/{note['asset_id']}/content")
    assert content.status_code == 200
    assert "practical baseline" in content.text


def test_edit_note_body_via_put(client, project):
    created = client.post(
        "/api/library/notes",
        params={"project_id": project.id},
        json={"title": "Editable", "content": "v1", "tags": ["x"]},
    ).json()

    resp = client.put(
        f"/api/library/notes/{created['asset_id']}",
        params={"project_id": project.id},
        json={"content": "# v2\n\nedited"},
    )
    assert resp.status_code == 200
    assert resp.json()["tags"] == ["x"]  # metadata preserved

    content = client.get(f"/api/assets/{created['asset_id']}/content")
    assert "edited" in content.text


def test_edit_unknown_note_404s(client, project):
    resp = client.put(
        "/api/library/notes/nope",
        params={"project_id": project.id},
        json={"content": "x"},
    )
    assert resp.status_code == 404


def test_add_and_delete_reference(client, project):
    resp = client.post(
        "/api/library/references",
        params={"project_id": project.id},
        json={
            "key": "so3krates2026",
            "title": "8-bit QAT of an SO(3)-equivariant transformer",
            "arxiv": "2601.02213",
            "tags": ["quantization"],
        },
    )
    assert resp.status_code == 201

    index = client.get("/api/library", params={"project_id": project.id}).json()
    assert index["references"][0]["key"] == "so3krates2026"

    assert (
        client.delete(
            "/api/library/references/so3krates2026", params={"project_id": project.id}
        ).status_code
        == 204
    )
    index = client.get("/api/library", params={"project_id": project.id}).json()
    assert index["references"] == []


def test_unknown_project_404s(client):
    resp = client.get("/api/library", params={"project_id": "does-not-exist"})
    assert resp.status_code == 404


def test_get_discovers_loose_readme(client, project):
    # A README dropped directly in the project dir shows up via GET (sync).
    (project.project_dir / "README.md").write_text("# Project readme\n\nhi\n")
    index = client.get("/api/library", params={"project_id": project.id}).json()
    titles = [n["title"] for n in index["notes"]]
    assert "Project readme" in titles
    discovered = next(n for n in index["notes"] if n["title"] == "Project readme")
    assert discovered["path"] == "README.md"
    # And its body is fetchable for preview.
    content = client.get(f"/api/assets/{discovered['asset_id']}/content")
    assert content.status_code == 200
    assert "hi" in content.text


def test_zotero_import_route(client, project, tmp_path):
    import sqlite3

    data_dir = tmp_path / "Zotero"
    data_dir.mkdir()
    db = data_dir / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INTEGER, key TEXT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
        CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, creatorTypeID INTEGER, orderIndex INTEGER);
        CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE itemTags (itemID INTEGER, tagID INTEGER);
        CREATE TABLE itemAttachments (itemID INTEGER, parentItemID INTEGER, path TEXT, contentType TEXT);
        CREATE TABLE deletedItems (itemID INTEGER);
        INSERT INTO itemTypes VALUES (1, 'journalArticle');
        INSERT INTO items VALUES (10, 1, 'KEY12345');
        INSERT INTO fields VALUES (1, 'title');
        INSERT INTO itemDataValues VALUES (1, 'A Zotero paper');
        INSERT INTO itemData VALUES (10, 1, 1);
        """
    )
    conn.commit()
    conn.close()

    resp = client.post(
        "/api/library/zotero/import",
        params={"project_id": project.id},
        json={"path": str(data_dir)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["imported"] == 1
    assert body["references"][0]["key"] == "KEY12345"
    assert body["references"][0]["source"] == "zotero"

    # Reference is now in the index, and the source is recorded.
    index = client.get("/api/library", params={"project_id": project.id}).json()
    assert index["references"][0]["title"] == "A Zotero paper"
    sources = client.get("/api/library/sources", params={"project_id": project.id}).json()
    assert sources[0]["kind"] == "zotero"


def test_zotero_import_bad_path_400s(client, project, tmp_path):
    resp = client.post(
        "/api/library/zotero/import",
        params={"project_id": project.id},
        json={"path": str(tmp_path / "nonexistent")},
    )
    assert resp.status_code == 400
