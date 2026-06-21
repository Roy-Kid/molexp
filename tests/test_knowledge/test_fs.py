"""Tests for the knowledge FileSystem seam (okf-05-04).

Folder/Library route all I/O through an injectable FileSystem. A dict-backed
FakeFileSystem proves the routing: with it injected, a whole bundle lives in
memory and the on-disk root stays empty.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from molexp.knowledge import (
    ConceptMeta,
    FileSystem,
    Folder,
    Library,
    LocalFileSystem,
    Workspace,
)


class FakeFileSystem:
    """In-memory FileSystem (no disk touched)."""

    def __init__(self) -> None:
        self.files: dict[Path, str] = {}
        self.dirs: set[Path] = set()

    def _touch_parents(self, path: Path) -> None:
        for ancestor in path.parents:
            self.dirs.add(ancestor)

    def read_text(self, path: Path) -> str:
        return self.files[path]

    def write_text(self, path: Path, content: str) -> None:
        self.files[path] = content
        self._touch_parents(path)

    def read_json(self, path: Path) -> Any:
        return json.loads(self.files[path])

    def write_json(self, path: Path, data: Any) -> None:
        self.files[path] = json.dumps(data)
        self._touch_parents(path)

    def exists(self, path: Path) -> bool:
        return path in self.files or path in self.dirs

    def is_file(self, path: Path) -> bool:
        return path in self.files

    def is_dir(self, path: Path) -> bool:
        return path in self.dirs

    def mkdir(self, path: Path) -> None:
        self.dirs.add(path)
        self._touch_parents(path)

    def iterdir(self, path: Path) -> list[Path]:
        seen = {p for p in (*self.files, *self.dirs) if p.parent == path}
        return sorted(seen)

    def rmtree(self, path: Path) -> None:
        self.files = {p: c for p, c in self.files.items() if path not in (p, *p.parents)}
        self.dirs = {p for p in self.dirs if path not in (p, *p.parents)}

    def lock(self, path: Path) -> Any:
        return nullcontext()


# ── surface + LocalFileSystem (ac-001 / ac-002) ──────────────────────────────


def test_surface_and_local_fs_round_trips(tmp_path: Path) -> None:
    assert isinstance(LocalFileSystem(), FileSystem)  # runtime_checkable Protocol
    fs = LocalFileSystem()
    d = tmp_path / "d"
    fs.mkdir(d)
    assert fs.is_dir(d)
    fs.write_text(d / "a.txt", "hello")
    assert fs.read_text(d / "a.txt") == "hello"
    fs.write_json(d / "a.json", {"k": 1})
    assert fs.read_json(d / "a.json") == {"k": 1}
    assert fs.is_file(d / "a.txt")
    assert {p.name for p in fs.iterdir(d)} == {"a.txt", "a.json"}
    with fs.lock(d / "x.lock"):
        pass
    fs.rmtree(d)
    assert not fs.exists(d)


# ── Folder routes through injected fs (ac-003 / ac-004) ───────────────────────


def test_folder_routes_through_injected_fs(tmp_path: Path) -> None:
    fake = FakeFileSystem()
    f = Folder(name="alpha", root=tmp_path, fs=fake)
    f.write_meta(ConceptMeta(type="run", id="r1"))
    f.write_index("# Alpha\n")
    f.write_ops_json("state", {"n": 1})

    # everything landed in the fake; nothing on real disk
    assert list(tmp_path.iterdir()) == []
    assert f.read_meta().id == "r1"
    assert f.read_index() == "# Alpha\n"
    assert f.read_ops_json("state") == {"n": 1}


def test_child_inherits_parent_fs(tmp_path: Path) -> None:
    fake = FakeFileSystem()
    root = Folder(name="bundle", root=tmp_path, fs=fake)
    child = root.add_folder("kid", concept_type="run")
    assert child.fs is fake
    child.write_index("body")
    assert list(tmp_path.iterdir()) == []  # nothing on disk
    assert root.get_folder("kid").read_index() == "body"


# ── full hierarchy + Library on the fake fs (ac-005) ─────────────────────────


def test_hierarchy_and_library_on_fake_fs(tmp_path: Path) -> None:
    fake = FakeFileSystem()
    ws = Workspace(name="lab", root=tmp_path, fs=fake)
    ws.add_project("p").add_experiment("e").add_run("r")

    lib = Library(tmp_path, fs=fake)
    rels = {lib.rel_path(c) for c in lib.walk()}
    assert rels == {"lab", "lab/p", "lab/p/e", "lab/p/e/r"}

    idx = lib.build_index()
    assert {e.path for e in idx.entries} == rels
    assert list(tmp_path.iterdir()) == []  # bundle lives entirely in the fake


def test_reference_read_ref_meta_routes_through_fs(tmp_path: Path) -> None:
    from molexp.knowledge import Reference, ReferenceMeta

    fake = FakeFileSystem()
    ref = Reference(name="smith2024", root=tmp_path, fs=fake)
    ref.write_ref_meta(ReferenceMeta(title="T", doi="10.1/x"))
    # read_ref_meta must come from the fake, not raw disk
    assert ref.read_ref_meta().title == "T"
    assert list(tmp_path.iterdir()) == []
