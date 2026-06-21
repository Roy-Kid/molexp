"""Tests for read-only Zotero import → Reference concepts on workspace (wsokf-05).

Mirrors ``tests/test_knowledge/test_zotero.py`` against the workspace surface. A
minimal ``zotero.sqlite`` is built in-place (the real schema subset the reader
touches), then imported via the OKF :class:`molexp.workspace.Bundle`. PDFs are
pointed at via ``pdf_path`` — no bytes are copied into the bundle.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from molexp.workspace import Bundle, ReferenceConcept, ZoteroItem, read_zotero_items

FIXED = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


def _make_zotero_db(data_dir: Path) -> Path:
    """Build a minimal zotero.sqlite + storage tree; return the db path."""
    db = data_dir / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INTEGER);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, orderIndex INTEGER);
        CREATE TABLE itemAttachments (
            itemID INTEGER, parentItemID INTEGER, path TEXT, contentType TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO itemTypes VALUES (?, ?)",
        [(1, "journalArticle"), (2, "attachment")],
    )
    conn.executemany(
        "INSERT INTO fields VALUES (?, ?)",
        [(1, "title"), (2, "DOI"), (3, "url"), (4, "date")],
    )
    conn.executemany(
        "INSERT INTO items VALUES (?, ?, ?)",
        [(10, "AAAA", 1), (11, "BBBB", 1), (20, "CCCC", 2)],
    )
    conn.executemany(
        "INSERT INTO itemDataValues VALUES (?, ?)",
        [
            (1, "Deep Learning"),
            (2, "10.1/x"),
            (3, "http://ex.com"),
            (4, "2015-05-01"),
            (5, "No PDF Paper"),
            (6, "2020"),
        ],
    )
    conn.executemany(
        "INSERT INTO itemData VALUES (?, ?, ?)",
        [
            (10, 1, 1),  # title
            (10, 2, 2),  # DOI
            (10, 3, 3),  # url
            (10, 4, 4),  # date
            (11, 1, 5),  # title
            (11, 4, 6),  # date
        ],
    )
    conn.executemany(
        "INSERT INTO creators VALUES (?, ?, ?)",
        [(100, "Yann", "LeCun")],
    )
    conn.executemany("INSERT INTO itemCreators VALUES (?, ?, ?)", [(10, 100, 0)])
    conn.executemany(
        "INSERT INTO itemAttachments VALUES (?, ?, ?, ?)",
        [(20, 10, "storage:paper.pdf", "application/pdf")],
    )
    conn.commit()
    conn.close()
    return db


# ── read_zotero_items (ac-006) ───────────────────────────────────────────────


def test_read_zotero_items_parses_fields(tmp_path: Path) -> None:
    db = _make_zotero_db(tmp_path)
    items = {i.key: i for i in read_zotero_items(db)}
    assert set(items) == {"AAAA", "BBBB"}  # attachment item excluded

    a = items["AAAA"]
    assert isinstance(a, ZoteroItem)
    assert a.title == "Deep Learning"
    assert a.authors == ("Yann LeCun",)
    assert a.year == 2015
    assert a.doi == "10.1/x"
    assert a.pdf_path is not None
    assert a.pdf_path.endswith("storage/CCCC/paper.pdf")

    assert items["BBBB"].year == 2020
    assert items["BBBB"].pdf_path is None


def test_read_zotero_items_does_not_modify_db(tmp_path: Path) -> None:
    db = _make_zotero_db(tmp_path)
    before = db.read_bytes()
    read_zotero_items(db)
    assert db.read_bytes() == before  # opened read-only


# ── Bundle.import_zotero (ac-007) ────────────────────────────────────────────


def test_import_zotero_creates_reference_pointers(tmp_path: Path) -> None:
    src = tmp_path / "zotero"
    src.mkdir()
    db = _make_zotero_db(src)
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    b = Bundle(bundle_root)

    refs = b.import_zotero(db, now=FIXED)
    assert all(isinstance(r, ReferenceConcept) for r in refs)
    by_key = {r.read_ref_meta().source_key: r for r in b.references()}
    assert set(by_key) == {"AAAA", "BBBB"}

    a_meta = by_key["AAAA"].read_ref_meta()
    assert a_meta.source == "zotero"
    assert a_meta.title == "Deep Learning"
    assert a_meta.pdf_path is not None and a_meta.pdf_path.endswith("storage/CCCC/paper.pdf")

    # no PDF bytes copied into the bundle
    assert list(bundle_root.rglob("*.pdf")) == []


def test_import_zotero_idempotent_on_source_key(tmp_path: Path) -> None:
    src = tmp_path / "zotero"
    src.mkdir()
    db = _make_zotero_db(src)
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    b = Bundle(bundle_root)

    b.import_zotero(db, now=FIXED)
    b.import_zotero(db, now=FIXED)  # re-import
    assert len(b.references()) == 2  # no duplicates


def test_import_zotero_records_sources_json(tmp_path: Path) -> None:
    import json

    src = tmp_path / "zotero"
    src.mkdir()
    db = _make_zotero_db(src)
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    b = Bundle(bundle_root)

    b.import_zotero(db, now=FIXED)
    sources = json.loads((bundle_root / "sources.json").read_text())
    assert len(sources) == 1
    entry = sources[0]
    assert entry["source"] == "zotero"
    assert entry["count"] == 2
    assert entry["imported_at"].startswith("2026-06-21T12:00:00")
    assert "+00:00" in entry["imported_at"]  # aware-UTC
