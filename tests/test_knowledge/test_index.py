"""Tests for the derived bundle index + search (okf-03).

``Library.build_index()`` rolls the whole Concept tree (``meta.yaml`` +
markdown-link graph) into a ``LibraryIndex`` and writes two *derived* siblings
at the bundle root — ``index.json`` (machine) + ``INDEX.md`` (human/agent).
``search()`` filters that index by type / tag / text. Neither file is a source
of truth; both are rebuilt on demand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.knowledge import (
    ConceptIndexEntry,
    Folder,
    Library,
    LibraryIndex,
    Workspace,
)
from molexp.knowledge.index import INDEX_JSON_FILENAME, INDEX_MD_FILENAME

FIXED = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


def _hierarchy(tmp_path: Path) -> Path:
    ws = Workspace(name="lab", root=tmp_path)
    ws.add_project("p").add_experiment("e").add_run("r")
    return tmp_path


# ── models (ac-002) ──────────────────────────────────────────────────────────


def test_entry_and_index_frozen() -> None:
    e = ConceptIndexEntry(path="a", type="folder")
    with pytest.raises(ValidationError):
        e.path = "b"  # type: ignore[misc]
    idx = LibraryIndex()
    with pytest.raises(ValidationError):
        idx.generated_at = FIXED  # type: ignore[misc]


def test_index_json_round_trip_and_markdown() -> None:
    idx = LibraryIndex(
        generated_at=FIXED,
        entries=(
            ConceptIndexEntry(path="lab", type="workspace", title="Lab", tags=("x",)),
            ConceptIndexEntry(path="lab/p", type="project", links=("lab",)),
        ),
    )
    back = LibraryIndex.model_validate(idx.model_dump(mode="json"))
    assert back == idx
    md = idx.to_markdown()
    assert "lab" in md and "lab/p" in md
    assert "index.json" in md  # points at the machine sibling


# ── build_index (ac-003 / ac-004 / ac-005) ───────────────────────────────────


def test_build_index_entries_equal_walk_set(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    lib = Library(root)
    idx = lib.build_index(now=FIXED)
    assert {e.path for e in idx.entries} == {lib.rel_path(f) for f in lib.walk()}
    by_path = {e.path: e for e in idx.entries}
    assert by_path["lab"].type == "workspace"
    assert by_path["lab/p/e/r"].type == "run"


def test_build_index_writes_derived_siblings(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    lib = Library(root)
    lib.build_index(now=FIXED)
    assert (root / INDEX_JSON_FILENAME).is_file()
    assert (root / INDEX_MD_FILENAME).is_file()
    md = (root / INDEX_MD_FILENAME).read_text()
    assert "lab/p/e/r" in md
    # the derived files are not mistaken for concepts
    assert all(not lib.rel_path(f).endswith(".json") for f in lib.walk())
    assert all(not lib.rel_path(f).endswith(".md") for f in lib.walk())


def test_entry_title_from_h1_else_name_and_links(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    a = root.add_folder("alpha")
    b = root.add_folder("beta")
    a.write_index("# Alpha Title\n\nbody\n- [to-b](../beta)\n")
    lib = Library(tmp_path)
    idx = lib.build_index(now=FIXED)
    by_path = {e.path: e for e in idx.entries}
    assert by_path["bundle/alpha"].title == "Alpha Title"
    assert by_path["bundle/beta"].title == "beta"  # no H1 → concept name
    assert by_path["bundle/beta"].path == lib.rel_path(b)
    # link resolves to beta as bundle-relative posix
    assert "bundle/beta" in by_path["bundle/alpha"].links


# ── search (ac-006) ──────────────────────────────────────────────────────────


def test_search_by_type(tmp_path: Path) -> None:
    lib = Library(_hierarchy(tmp_path))
    runs = lib.search(concept_type="run")
    assert [e.path for e in runs] == ["lab/p/e/r"]


def test_search_by_tag(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    tagged = root.add_folder("tagged")
    from molexp.knowledge import ConceptMeta

    tagged.write_meta(ConceptMeta(type="folder", tags=["important"]))
    root.add_folder("plain")
    lib = Library(tmp_path)
    hits = lib.search(tag="important")
    assert [e.path for e in hits] == ["bundle/tagged"]


def test_search_by_text_and_and_semantics(tmp_path: Path) -> None:
    lib = Library(_hierarchy(tmp_path))
    assert {e.path for e in lib.search("p")} >= {"lab/p", "lab/p/e", "lab/p/e/r"}
    # AND: text + type
    assert [e.path for e in lib.search("lab", concept_type="run")] == ["lab/p/e/r"]


def test_search_rebuild_reflects_new_concept(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    lib = Library(root)
    lib.build_index(now=FIXED)
    Workspace(name="lab", root=root).add_project("q")
    assert any(e.path == "lab/q" for e in lib.search(rebuild=True))


# ── derived / rebuildable (ac-007) ───────────────────────────────────────────


def test_build_index_idempotent_same_now(tmp_path: Path) -> None:
    lib = Library(_hierarchy(tmp_path))
    assert lib.build_index(now=FIXED) == lib.build_index(now=FIXED)


def test_generated_at_is_aware_utc(tmp_path: Path) -> None:
    lib = Library(_hierarchy(tmp_path))
    idx = lib.build_index()
    assert idx.generated_at is not None
    assert idx.generated_at.tzinfo is not None  # aware


def test_rebuild_restores_deleted_siblings(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    lib = Library(root)
    lib.build_index(now=FIXED)
    (root / INDEX_JSON_FILENAME).unlink()
    (root / INDEX_MD_FILENAME).unlink()
    lib.build_index(now=FIXED)
    assert (root / INDEX_JSON_FILENAME).is_file()
    assert (root / INDEX_MD_FILENAME).is_file()
