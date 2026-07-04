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


# ── vision-loop-08 — ``molexp knowledge search`` ──────────────────────────────
#
# The command is PURE EXPOSURE of ``Bundle.search`` (the same verb the server
# route wraps — one function, three faces): these tests pin the CLI face only
# (workspace resolution via --path, table rendering, q/--type/--tag plumbing).
# Category map (tester):
#     basics       — command registered; matching paths rendered
#     edge cases   — no match exits 0 with no result rows
#     integration  — seeded Note concepts on disk -> CLI round-trip

# Wide virtual terminal so rich never wraps a path/snippet cell mid-token.
_WIDE_TERM = {"COLUMNS": "200"}

# Needle that only ever appears in a seeded note body — never in a path/title.
_CLI_BODY_NEEDLE = "vl08-cli-needle"


def _seed_search_notes(ws_root: Path) -> None:
    """Two root-level notes: ``alpha-note`` (tagged, body needle) + ``beta-note``."""
    from molexp.workspace import Workspace
    from molexp.workspace.concepts import Note

    ws = Workspace.load(ws_root)
    alpha = ws.add_folder(Note(parent=ws, name="alpha-note"))
    alpha.set_body(f"# Alpha Note\n\nthe {_CLI_BODY_NEEDLE} sits here\n")
    alpha.set_tags(["vl08tag"])
    beta = ws.add_folder(Note(parent=ws, name="beta-note"))
    beta.set_body("# Beta Note\n\nnothing to see\n")


@pytest.mark.integration
def test_knowledge_search_prints_matching_paths(runner, initialized_ws):
    """A body-needle query renders the matching note's path — and only it."""
    _seed_search_notes(initialized_ws)

    result = runner.invoke(
        app,
        ["knowledge", "search", _CLI_BODY_NEEDLE, "--path", str(initialized_ws)],
        env=_WIDE_TERM,
    )
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "alpha-note" in out
    assert "beta-note" not in out


@pytest.mark.integration
def test_knowledge_search_tag_filter_narrows(runner, initialized_ws):
    """--tag forwards to Bundle.search: both notes match the query, one the tag."""
    _seed_search_notes(initialized_ws)

    result = runner.invoke(
        app,
        ["knowledge", "search", "note", "--tag", "vl08tag", "--path", str(initialized_ws)],
        env=_WIDE_TERM,
    )
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "alpha-note" in out
    assert "beta-note" not in out


@pytest.mark.integration
def test_knowledge_search_type_filter_narrows(runner, initialized_ws):
    """--type forwards to Bundle.search: a note never matches a reference type."""
    _seed_search_notes(initialized_ws)

    result = runner.invoke(
        app,
        [
            "knowledge",
            "search",
            "alpha",
            "--type",
            "reference.reference",
            "--path",
            str(initialized_ws),
        ],
        env=_WIDE_TERM,
    )
    assert result.exit_code == 0, result.output
    assert "alpha-note" not in _flat(result.output)
