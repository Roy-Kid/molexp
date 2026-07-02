"""Tests for ``molexp knowledge import-zotero``.

The command is pure plumbing over the existing workspace surface:
:func:`molexp.workspace.read_zotero_items` (via :meth:`Bundle.import_zotero`)
materializes each Zotero item as a ``ReferenceConcept`` directory under the
destination workspace — PDFs pointed at, never copied. A minimal
``zotero.sqlite`` fixture (the schema subset the reader touches) is built
in-place, mirroring ``tests/test_workspace/test_zotero_concepts.py``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from molexp.cli import app


def _plain(text: str) -> str:
    """Strip ANSI colour codes so substring asserts survive rich styling."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _flat(text: str) -> str:
    """Collapse all whitespace so substring asserts survive rich line-wrapping."""
    return " ".join(_plain(text).split())


def _make_zotero_db(data_dir: Path) -> Path:
    """Build a minimal zotero.sqlite + storage tree; return the db path."""
    data_dir.mkdir(parents=True, exist_ok=True)
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
    conn.executemany("INSERT INTO fields VALUES (?, ?)", [(1, "title"), (4, "date")])
    conn.executemany(
        "INSERT INTO items VALUES (?, ?, ?)",
        [(10, "AAAA", 1), (11, "BBBB", 1), (20, "CCCC", 2)],
    )
    conn.executemany(
        "INSERT INTO itemDataValues VALUES (?, ?)",
        [(1, "Deep Learning"), (2, "2015-05-01"), (3, "No PDF Paper"), (4, "2020")],
    )
    conn.executemany(
        "INSERT INTO itemData VALUES (?, ?, ?)",
        [(10, 1, 1), (10, 4, 2), (11, 1, 3), (11, 4, 4)],
    )
    conn.executemany(
        "INSERT INTO itemAttachments VALUES (?, ?, ?, ?)",
        [(20, 10, "storage:paper.pdf", "application/pdf")],
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def initialized_ws(tmp_path, runner):
    """A freshly initialised workspace at ``tmp_path / 'lab'``."""
    ws_root = tmp_path / "lab"
    runner.invoke(app, ["init", str(ws_root)])
    return ws_root


@pytest.fixture
def zotero_db(tmp_path):
    return _make_zotero_db(tmp_path / "zotero-data")


# ── surface ───────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_knowledge_group_registered(runner):
    result = runner.invoke(app, ["knowledge", "--help"])
    assert result.exit_code == 0
    assert "import-zotero" in _plain(result.stdout)


# ── happy path ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_import_zotero_materializes_reference_concepts(runner, initialized_ws, zotero_db):
    result = runner.invoke(
        app,
        ["knowledge", "import-zotero", str(zotero_db), "--dest", str(initialized_ws)],
    )
    assert result.exit_code == 0, result.output
    out = _plain(result.stdout)
    assert "Imported 2 reference(s)" in out
    assert "Deep Learning" in out

    # Each item is a ReferenceConcept directory (meta.yaml marks the type).
    ref_dir = initialized_ws / "references" / "aaaa"
    assert (ref_dir / "meta.yaml").is_file()
    assert "reference.reference" in (ref_dir / "meta.yaml").read_text()

    # PDFs are pointed at, never copied into the workspace.
    assert list(initialized_ws.rglob("*.pdf")) == []


@pytest.mark.integration
def test_import_zotero_accepts_data_directory(runner, initialized_ws, zotero_db):
    """Passing the Zotero data directory finds zotero.sqlite inside it."""
    result = runner.invoke(
        app,
        ["knowledge", "import-zotero", str(zotero_db.parent), "--dest", str(initialized_ws)],
    )
    assert result.exit_code == 0, result.output
    assert "Imported 2 reference(s)" in _plain(result.stdout)


@pytest.mark.integration
def test_import_zotero_is_idempotent(runner, initialized_ws, zotero_db):
    args = ["knowledge", "import-zotero", str(zotero_db), "--dest", str(initialized_ws)]
    assert runner.invoke(app, args).exit_code == 0
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output

    refs_root = initialized_ws / "references"
    ref_dirs = [p for p in refs_root.iterdir() if p.is_dir()]
    assert len(ref_dirs) == 2  # no duplicates on re-import


# ── human-readable failure modes ─────────────────────────────────────────────


@pytest.mark.integration
def test_import_zotero_missing_database_errors(runner, initialized_ws, tmp_path):
    result = runner.invoke(
        app,
        [
            "knowledge",
            "import-zotero",
            str(tmp_path / "nowhere" / "zotero.sqlite"),
            "--dest",
            str(initialized_ws),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in _flat(result.output)


@pytest.mark.integration
def test_import_zotero_non_sqlite_file_errors(runner, initialized_ws, tmp_path):
    bogus = tmp_path / "zotero.sqlite"
    bogus.write_text("this is not a sqlite database")
    result = runner.invoke(
        app,
        ["knowledge", "import-zotero", str(bogus), "--dest", str(initialized_ws)],
    )
    assert result.exit_code == 1
    assert "not a Zotero database" in _flat(result.output)


@pytest.mark.integration
def test_import_zotero_sqlite_without_zotero_tables_errors(runner, initialized_ws, tmp_path):
    other = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()
    result = runner.invoke(
        app,
        ["knowledge", "import-zotero", str(other), "--dest", str(initialized_ws)],
    )
    assert result.exit_code == 1
    assert "not a Zotero database" in _flat(result.output)


@pytest.mark.integration
def test_import_zotero_locked_database_says_close_zotero(
    runner, initialized_ws, zotero_db, monkeypatch
):
    """A running Zotero holds an exclusive lock — the error must say so."""

    def _locked(path):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("molexp.workspace.bundle.read_zotero_items", _locked)
    result = runner.invoke(
        app,
        ["knowledge", "import-zotero", str(zotero_db), "--dest", str(initialized_ws)],
    )
    assert result.exit_code == 1
    assert "close Zotero" in _flat(result.output)


@pytest.mark.integration
def test_import_zotero_dest_not_a_workspace_errors(runner, zotero_db, tmp_path):
    not_ws = tmp_path / "plain-dir"
    not_ws.mkdir()
    result = runner.invoke(
        app,
        ["knowledge", "import-zotero", str(zotero_db), "--dest", str(not_ws)],
    )
    assert result.exit_code == 1
    out = _plain(result.output)
    assert "not a molexp workspace" in out
    assert "molexp init" in out  # actionable hint
