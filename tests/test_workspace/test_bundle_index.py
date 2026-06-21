"""Tests for the derived bundle index + search.

``Bundle.build_index()`` rolls the whole Concept tree (``meta.yaml`` +
markdown-link graph) into a :class:`BundleIndex` and writes two *derived*
siblings at the bundle root — ``index.json`` (machine) + ``INDEX.md``
(human/agent). ``search()`` filters that index by type / tag / text. Neither
file is a source of truth; both are rebuilt on demand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.workspace import Bundle, BundleIndex, ConceptIndexEntry, Folder, Workspace
from molexp.workspace.bundle_index import INDEX_JSON_FILENAME, INDEX_MD_FILENAME

FIXED = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)

CONCEPT_KIND = "bundle.concept"


def _hierarchy(tmp_path: Path) -> Path:
    ws = Workspace(root=tmp_path / "lab")
    ws.materialize()
    ws.add_project("p").add_experiment("e").add_run(id="r")
    return tmp_path


def _concept(name: str, root_path: Path) -> Folder:
    folder = Folder(name=name, kind=CONCEPT_KIND, root_path=str(root_path))
    folder.materialize()
    folder.write_meta()
    return folder


# ── models ───────────────────────────────────────────────────────────────────


def test_entry_and_index_frozen() -> None:
    e = ConceptIndexEntry(path="a", type="folder")
    with pytest.raises(ValidationError):
        e.path = "b"  # ty: ignore[invalid-assignment]
    idx = BundleIndex()
    with pytest.raises(ValidationError):
        idx.generated_at = FIXED  # ty: ignore[invalid-assignment]


def test_index_json_round_trip_and_markdown() -> None:
    idx = BundleIndex(
        generated_at=FIXED,
        entries=(
            ConceptIndexEntry(path="lab", type="workspace.root", title="Lab", tags=("x",)),
            ConceptIndexEntry(path="lab/p", type="workspace.project", links=("lab",)),
        ),
    )
    back = BundleIndex.model_validate(idx.model_dump(mode="json"))
    assert back == idx
    md = idx.to_markdown()
    assert "lab" in md and "lab/p" in md
    assert "index.json" in md  # points at the machine sibling


# ── build_index (ac-008) ──────────────────────────────────────────────────────


def test_build_index_entries_equal_walk_set(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    b = Bundle(root)
    idx = b.build_index(now=FIXED)
    assert {e.path for e in idx.entries} == {b.rel_path(f) for f in b.walk()}
    by_path = {e.path: e for e in idx.entries}
    assert by_path["lab"].type == "workspace.root"
    assert by_path["lab/projects/p/experiments/e/runs/run-r"].type == "workspace.run"


def test_build_index_writes_derived_siblings(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    b = Bundle(root)
    b.build_index(now=FIXED)
    assert (root / INDEX_JSON_FILENAME).is_file()
    assert (root / INDEX_MD_FILENAME).is_file()
    md = (root / INDEX_MD_FILENAME).read_text()
    assert "lab/projects/p" in md
    # the derived files are not mistaken for concepts
    assert all(not b.rel_path(f).endswith(".json") for f in b.walk())
    assert all(not b.rel_path(f).endswith(".md") for f in b.walk())


def test_build_index_title_from_h1_else_name_and_links(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    a = _concept("alpha", root)
    b_concept = _concept("beta", root)
    a.write_index("# Alpha Title\n\nbody\n- [to-b](../beta)\n")
    bundle = Bundle(root)
    idx = bundle.build_index(now=FIXED)
    by_path = {e.path: e for e in idx.entries}
    assert by_path["alpha"].title == "Alpha Title"
    assert by_path["beta"].title == "beta"  # no H1 → concept name
    assert by_path["beta"].path == bundle.rel_path(b_concept)
    # link resolves to beta as bundle-relative posix
    assert "beta" in by_path["alpha"].links


# ── search (ac-009) ───────────────────────────────────────────────────────────


def test_search_by_type(tmp_path: Path) -> None:
    b = Bundle(_hierarchy(tmp_path))
    runs = b.search(concept_type="workspace.run")
    assert [e.path for e in runs] == ["lab/projects/p/experiments/e/runs/run-r"]


def test_search_by_tag(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    tagged = _concept("tagged", root)
    # tags live in meta.yaml; write_meta() only stores type+id, so add tags directly
    (Path(tagged.resolve()) / "meta.yaml").write_text(
        "type: bundle.concept\nid: tagged\ntags:\n- important\n"
    )
    _concept("plain", root)
    b = Bundle(root)
    hits = b.search(tag="important")
    assert [e.path for e in hits] == ["tagged"]


def test_search_by_text_and_and_semantics(tmp_path: Path) -> None:
    b = Bundle(_hierarchy(tmp_path))
    paths = {e.path for e in b.search("p")}
    assert {"lab/projects/p", "lab/projects/p/experiments/e"} <= paths
    # AND: text + type
    assert [e.path for e in b.search("lab", concept_type="workspace.run")] == [
        "lab/projects/p/experiments/e/runs/run-r"
    ]


def test_search_rebuild_reflects_new_concept(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    b = Bundle(root)
    b.build_index(now=FIXED)
    Workspace(root=root / "lab").add_project("q")
    assert any(e.path == "lab/projects/q" for e in b.search(rebuild=True))


# ── derived / rebuildable (ac-008) ────────────────────────────────────────────


def test_build_index_idempotent_same_now(tmp_path: Path) -> None:
    b = Bundle(_hierarchy(tmp_path))
    assert b.build_index(now=FIXED) == b.build_index(now=FIXED)


def test_generated_at_is_aware_utc(tmp_path: Path) -> None:
    b = Bundle(_hierarchy(tmp_path))
    idx = b.build_index()
    assert idx.generated_at is not None
    assert idx.generated_at.tzinfo is not None  # aware


def test_rebuild_restores_deleted_siblings(tmp_path: Path) -> None:
    root = _hierarchy(tmp_path)
    b = Bundle(root)
    b.build_index(now=FIXED)
    (root / INDEX_JSON_FILENAME).unlink()
    (root / INDEX_MD_FILENAME).unlink()
    b.build_index(now=FIXED)
    assert (root / INDEX_JSON_FILENAME).is_file()
    assert (root / INDEX_MD_FILENAME).is_file()
