"""RED tests for the ``Bundle`` docs/notes CRUD surface (spec knowledge-docs-01-crud).

Covers six NEW public methods on :class:`molexp.workspace.Bundle`
(``create_note`` / ``rename_note`` / ``move_note`` / ``delete_note`` /
``backlinks`` / ``export_markdown``) plus a new :class:`Backlink`
``typing.NamedTuple``. None of these exist yet, so this module fails at import
(collection) time until the production symbols land — that is the desired RED
state (write-mode TDD).

Conventions mirror the sibling ``test_bundle.py``: ``from __future__ import
annotations``, build bundles on ``tmp_path``, and assert against
``Bundle.rel_path(...)`` path-as-identity rather than raw ``resolve()`` strings.

Acceptance map:
    ac-001  create_note — slug + meta + body, idempotency, child nesting
    ac-002  atomic writes only (recording FileSystem + source scan)
    ac-003  rename / move fidelity (body + child docs + backlinks)
    ac-004  delete
    ac-005  backlinks typed (roles, exclusions, no reverse-index file)
    ac-006  export_markdown (children in document order; flag off)
    ac-007  public surface (Backlink export + Bundle method presence)
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pytest

import molexp.workspace
import molexp.workspace.bundle as bundle_module
from molexp.workspace import (
    Backlink,
    Bundle,
    ConceptNotFoundError,
    Folder,
    Note,
)
from molexp.workspace.fs import PathArg
from molexp.workspace.fs_local import LocalFileSystem


@pytest.fixture
def bundle_root(tmp_path: Path) -> Path:
    """An empty on-disk bundle root (no concepts yet)."""
    root = tmp_path / "bundle"
    root.mkdir()
    return root


# ── ac-002 support: a write-path-recording FileSystem ────────────────────────


class RecordingFileSystem(LocalFileSystem):
    """A ``LocalFileSystem`` that logs every write entry point before delegating.

    Overriding the atomic writers + ``open`` lets a test prove a ``Note``'s body
    and ``meta.yaml`` were written through the *atomic* writers, and that NO
    write-mode bare ``open()`` was used for any file.
    """

    def __init__(self) -> None:
        self.atomic_text_writes: list[str] = []
        self.atomic_json_writes: list[str] = []
        self.write_opens: list[tuple[str, str]] = []

    def atomic_write_text(self, path: PathArg, content: str, *, encoding: str = "utf-8") -> None:
        self.atomic_text_writes.append(str(path))
        super().atomic_write_text(path, content, encoding=encoding)

    def atomic_write_json(self, path: PathArg, data: object) -> None:
        self.atomic_json_writes.append(str(path))
        super().atomic_write_json(path, data)

    def open(self, path: PathArg, mode: str = "r", encoding: str = "utf-8") -> IO[str]:
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            self.write_opens.append((str(path), mode))
        return super().open(path, mode=mode, encoding=encoding)


# ── ac-001 create_note: slug + meta + body, idempotency, child nesting ───────


def test_create_note_slugifies_dir_and_writes_meta_and_body(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    note = b.create_note("Design Doc", body="hi")

    assert isinstance(note, Note)
    assert b.rel_path(note) == "design-doc"
    assert (bundle_root / "design-doc").is_dir()
    assert note.read_meta()["type"] == "note.note"
    assert note.body() == "hi"


def test_create_note_is_idempotent_on_slug(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    first = b.create_note("Design Doc", body="hi")
    count_before = len(list(b.walk()))

    second = b.create_note("Design Doc", body="hi")

    assert b.rel_path(second) == b.rel_path(first) == "design-doc"
    assert len(list(b.walk())) == count_before
    rels = [b.rel_path(f) for f in b.walk()]
    assert rels.count("design-doc") == 1


def test_create_note_child_nests_under_parent(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    parent = b.create_note("Parent")
    child = b.create_note("Child", parent=parent)

    assert b.rel_path(parent) == "parent"
    assert b.rel_path(child) == "parent/child"
    assert (bundle_root / "parent" / "child").is_dir()
    assert b.get("parent/child").read_meta()["type"] == "note.note"


# ── ac-002 atomic writes only ────────────────────────────────────────────────


def test_create_note_writes_body_and_meta_through_atomic_writers(bundle_root: Path) -> None:
    rec = RecordingFileSystem()
    b = Bundle(bundle_root, fs=rec)

    note = b.create_note("Design Doc", body="hello world")

    assert note.body() == "hello world"
    assert any(p.endswith("design-doc/index.md") for p in rec.atomic_text_writes)
    assert any(p.endswith("design-doc/meta.yaml") for p in rec.atomic_text_writes)


def test_create_note_uses_no_write_mode_bare_open(bundle_root: Path) -> None:
    rec = RecordingFileSystem()
    b = Bundle(bundle_root, fs=rec)

    b.create_note("Design Doc", body="hello world")

    assert rec.write_opens == []


def test_bundle_module_does_not_top_level_import_atomic_write_json() -> None:
    # The docs verbs must route writes through ``self._fs`` (the injected
    # FileSystem), never a module-level ``atomic_write_json`` that bypasses it.
    source = Path(bundle_module.__file__).read_text(encoding="utf-8")
    assert "import atomic_write_json" not in source


# ── ac-003 rename / move fidelity ────────────────────────────────────────────


def test_rename_note_preserves_body_and_resolves_at_new_identity(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    note = b.create_note("Design Doc", body="ORIGINAL BODY")

    b.rename_note(note, "Renamed")

    renamed = b.get("renamed")
    assert renamed.read_meta()["type"] == "note.note"
    assert renamed.read_index() == "ORIGINAL BODY"
    with pytest.raises(ConceptNotFoundError):
        b.get("design-doc")


def test_rename_note_preserves_child_docs(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    parent = b.create_note("Parent", body="P")
    b.create_note("Child", parent=parent, body="C")

    b.rename_note(parent, "Renamed")

    assert b.get("renamed/child").read_index() == "C"


def test_move_note_resolves_under_new_parent(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    home = b.create_note("Home", body="H")
    note = b.create_note("Roamer", body="R")

    b.move_note(note, home)

    assert b.get("home/roamer").read_index() == "R"
    with pytest.raises(ConceptNotFoundError):
        b.get("roamer")


def test_move_note_keeps_backlinks_correct_from_new_location(bundle_root: Path) -> None:
    # move_note relocates the subtree verbatim — it does NOT rewrite existing
    # relative links inside index.md (spec out-of-scope). backlinks() is a
    # derived recompute over current on-disk state, so a link written from the
    # note's NEW location must surface under its new identity.
    b = Bundle(bundle_root)
    target = b.create_note("Target", body="T")
    home = b.create_note("Home", body="H")
    roamer = b.create_note("Roamer", body="R")

    b.move_note(roamer, home)  # roamer now resolves under home/roamer
    b.link(roamer, target)  # edge written relative to the new location

    sources = {b.rel_path(bl.source) for bl in b.backlinks(target)}
    assert "home/roamer" in sources


# ── ac-004 delete ────────────────────────────────────────────────────────────


def test_delete_note_removes_dir_and_walk_drops_it(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    note = b.create_note("Design Doc", body="hi")
    note_dir = bundle_root / "design-doc"
    assert note_dir.is_dir()

    b.delete_note(note)

    assert not note_dir.exists()
    assert "design-doc" not in {b.rel_path(f) for f in b.walk()}


# ── ac-005 backlinks typed ───────────────────────────────────────────────────


def test_backlinks_returns_typed_sources_excluding_unrelated(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    dst = b.create_note("Target", body="T")
    src1 = b.create_note("Src One", body="1")
    src2 = b.create_note("Src Two", body="2")
    other = b.create_note("Unrelated", body="U")

    b.link(src1, dst, role="cites")
    b.link(src2, dst)  # default role == references
    b.link(other, src1)  # unrelated edge — must NOT surface as a backlink of dst

    got = {(b.rel_path(bl.source), bl.role) for bl in b.backlinks(dst)}
    assert got == {("src-one", "cites"), ("src-two", "references")}
    assert all(isinstance(bl.source, Folder) for bl in b.backlinks(dst))


def test_backlinks_persists_no_reverse_index_file(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    dst = b.create_note("Target", body="T")
    src = b.create_note("Src One", body="1")
    b.link(src, dst, role="cites")

    before = {p.name for p in bundle_root.iterdir()}
    _ = b.backlinks(dst)
    after = {p.name for p in bundle_root.iterdir()}

    assert before == after
    assert not (bundle_root / "backlinks.json").exists()


# ── ac-006 export_markdown ───────────────────────────────────────────────────


def test_export_markdown_includes_children_in_document_order(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    note = b.create_note("Parent", body="PARENT")
    b.create_note("Child", parent=note, body="CHILD")

    out = b.export_markdown(note, include_children=True)

    assert isinstance(out, str)
    assert "PARENT" in out
    assert "CHILD" in out
    assert out.index("PARENT") < out.index("CHILD")


def test_export_markdown_headers_descendants_by_bundle_relative_path(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    note = b.create_note("Parent", body="PARENT")
    b.create_note("Child", parent=note, body="CHILD")

    out = b.export_markdown(note, include_children=True)

    assert "## parent/child" in out


def test_export_markdown_excludes_children_when_flag_false(bundle_root: Path) -> None:
    b = Bundle(bundle_root)
    note = b.create_note("Parent", body="PARENT")
    b.create_note("Child", parent=note, body="CHILD")

    out = b.export_markdown(note, include_children=False)

    assert "PARENT" in out
    assert "CHILD" not in out


# ── ac-007 public surface ────────────────────────────────────────────────────


def test_backlink_importable_from_workspace() -> None:
    from molexp.workspace import Backlink as ImportedBacklink

    assert ImportedBacklink is Backlink


def test_backlink_exported_on_both_all_lists() -> None:
    assert "Backlink" in molexp.workspace.__all__
    assert "Backlink" in bundle_module.__all__
    assert molexp.workspace.Backlink is Backlink


def test_backlink_is_named_tuple_with_source_and_role() -> None:
    assert issubclass(Backlink, tuple)
    assert Backlink._fields == ("source", "role")


def test_bundle_exposes_doc_methods() -> None:
    for name in (
        "create_note",
        "rename_note",
        "move_note",
        "delete_note",
        "backlinks",
        "export_markdown",
    ):
        assert callable(getattr(Bundle, name)), name
