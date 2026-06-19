"""Zotero link/import — reads a local zotero.sqlite into References."""

from __future__ import annotations

import sqlite3

import pytest

from molexp.workspace.library.zotero import ZoteroImportError, read_zotero_references


def _build_zotero_db(data_dir):
    """Create a minimal Zotero-shaped SQLite DB with one journal article."""
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
        """
    )
    conn.executemany(
        "INSERT INTO itemTypes VALUES (?, ?)",
        [(1, "journalArticle"), (2, "attachment")],
    )
    conn.executemany(
        "INSERT INTO fields VALUES (?, ?)",
        [(1, "title"), (2, "date"), (3, "DOI"), (4, "url"), (5, "extra"), (6, "abstractNote")],
    )
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")
    conn.executemany("INSERT INTO creators VALUES (?, ?, ?)", [(1, "Ada", "Lovelace")])
    # Article item (key ABCD1234) + its PDF attachment (key PDF99999).
    conn.executemany(
        "INSERT INTO items VALUES (?, ?, ?)",
        [(10, 1, "ABCD1234"), (11, 2, "PDF99999")],
    )
    conn.executemany(
        "INSERT INTO itemDataValues VALUES (?, ?)",
        [
            (1, "Noisy forces as an effective temperature"),
            (2, "2026-03-01"),
            (3, "10.1000/xyz"),
            (4, "https://arxiv.org/abs/2601.02213"),
            (5, "arXiv: 2601.02213"),
            (6, "An abstract about ΔF."),
        ],
    )
    conn.executemany(
        "INSERT INTO itemData VALUES (?, ?, ?)",
        [(10, 1, 1), (10, 2, 2), (10, 3, 3), (10, 4, 4), (10, 5, 5), (10, 6, 6)],
    )
    conn.execute("INSERT INTO itemCreators VALUES (10, 1, 1, 0)")
    conn.execute("INSERT INTO tags VALUES (1, 'quantization')")
    conn.execute("INSERT INTO itemTags VALUES (10, 1)")
    conn.execute(
        "INSERT INTO itemAttachments VALUES (11, 10, 'storage:paper.pdf', 'application/pdf')"
    )
    conn.commit()
    conn.close()
    # The pointed-at PDF lives under storage/<attachment key>/.
    pdf_dir = data_dir / "storage" / "PDF99999"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    return db


def test_read_zotero_references_maps_fields(tmp_path):
    data_dir = tmp_path / "Zotero"
    data_dir.mkdir()
    _build_zotero_db(data_dir)

    refs = read_zotero_references(data_dir)  # accepts the data dir
    assert len(refs) == 1
    ref = refs[0]
    assert ref.key == "ABCD1234"
    assert ref.title == "Noisy forces as an effective temperature"
    assert ref.authors == ("Ada Lovelace",)
    assert ref.year == 2026
    assert ref.doi == "10.1000/xyz"
    assert ref.arxiv == "2601.02213"
    assert ref.tags == ("quantization",)
    assert ref.source == "zotero"
    assert ref.source_key == "ABCD1234"
    # PDF is pointed at (not copied) under the Zotero storage tree.
    assert ref.pdf_path is not None
    assert ref.pdf_path.endswith("storage/PDF99999/paper.pdf")


def test_read_zotero_accepts_sqlite_file_directly(tmp_path):
    data_dir = tmp_path / "Zotero"
    data_dir.mkdir()
    db = _build_zotero_db(data_dir)
    assert len(read_zotero_references(db)) == 1


def test_missing_library_raises(tmp_path):
    with pytest.raises(ZoteroImportError):
        read_zotero_references(tmp_path / "nope")


def test_import_zotero_into_library(project, tmp_path):
    data_dir = tmp_path / "Zotero"
    data_dir.mkdir()
    _build_zotero_db(data_dir)

    refs = project.library.import_zotero(data_dir)
    assert len(refs) == 1
    assert project.library.list_references()[0].key == "ABCD1234"

    sources = project.library.list_sources()
    assert sources[0]["kind"] == "zotero"
    assert sources[0]["count"] == 1
