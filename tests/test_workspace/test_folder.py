"""Tests for ``molexp.workspace.folder`` — the abstract ``Folder`` base class.

Covers the ``Folder`` lifecycle (lazy mkdir, atomic ``write_json``, id/kind
validation, ``children`` filtering, metadata round-trip, ``delete`` / ``move_to``)
and the typed markdown-graph edges (``append_link`` / ``typed_out_edges``).
The folder-module import-guard subprocess lives at the bottom.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from molexp.workspace.base import _load_metadata
from molexp.workspace.folder import Folder, FolderMoveCollisionError, append_link
from molexp.workspace.models import FolderMetadata


# ``Folder`` has no business subclasses at this level; this private subclass
# exists only so ``children()`` has something concrete to reconstruct.
class _TestSubFolder(Folder):
    """Minimal Folder subclass used only by the children() filter test."""


class TestFolder:
    def test_construction_is_side_effect_free_then_path_lazily_mkdirs(self, tmp_path: Path) -> None:
        """Construction touches no filesystem; the first ``path()`` mkdirs, and
        a second ``path()`` is an idempotent no-op returning the same Path."""
        folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)

        target = tmp_path / "alpha"
        assert not target.exists(), "construction must be side-effect-free"

        first = Path(folder.path())
        assert first == target
        assert first.is_dir()
        assert Path(folder.path()) == first  # idempotent

    def test_write_json_round_trips_and_survives_mid_write_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path round-trips; a mid-write ``os.replace`` failure leaves the
        pre-existing file intact (atomic temp-file + rename)."""
        folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)

        written = Path(str(folder.write_json("data.json", {"k": 1})))
        assert json.loads(written.read_text()) == {"k": 1}

        target_path = Path(folder.path()) / "data.json"

        def _explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated mid-write failure")

        monkeypatch.setattr("molexp.atomicio.os.replace", _explode)
        with pytest.raises(RuntimeError):
            folder.write_json("data.json", {"k": 2})

        assert json.loads(target_path.read_text()) == {"k": 1}

    @pytest.mark.parametrize(
        "name, valid",
        [
            ("ok-1", True),
            ("UPPER", False),  # uppercase rejected
            ("../etc", False),  # path traversal rejected
            ("", False),  # empty rejected
            ("!!!", False),  # all-punctuation slugifies to empty
        ],
    )
    def test_name_is_slugified_and_validated(self, tmp_path: Path, name: str, valid: bool) -> None:
        """``slugify(name)`` must yield an id matching ``_KIND_PATTERN``;
        otherwise ``ValueError`` at construction."""
        if valid:
            Folder(parent=None, name=name, kind="test.root", root_path=tmp_path)
        else:
            with pytest.raises(ValueError):
                Folder(parent=None, name=name, kind="test.root", root_path=tmp_path)

    @pytest.mark.parametrize(
        "kind, valid",
        [
            ("workspace.project", True),
            ("WORKSPACE.foo", False),  # uppercase rejected
            (".leading", False),  # leading dot rejected
            ("../etc", False),  # path traversal rejected
            ("", False),  # empty rejected
            ("foo bar", False),  # invalid char rejected
        ],
    )
    def test_kind_is_validated(self, tmp_path: Path, kind: str, valid: bool) -> None:
        """``kind`` must be dotted lowercase ASCII (``_KIND_PATTERN``)."""
        if valid:
            Folder(parent=None, name="alpha", kind=kind, root_path=tmp_path)
        else:
            with pytest.raises(ValueError):
                Folder(parent=None, name="alpha", kind=kind, root_path=tmp_path)

    def test_construction_shapes_parent_vs_root_path(self, tmp_path: Path) -> None:
        """``parent=None`` + ``root_path=None`` is the unmounted state (legal at
        construction; ``.path()`` raises until mounted). ``parent`` + ``root_path``
        both set is a ``ValueError``. Nesting walks parent→child correctly."""
        unmounted = Folder(parent=None, name="alpha", kind="test.root", root_path=None)
        assert unmounted._parent is None
        with pytest.raises(RuntimeError, match="unmounted"):
            Path(unmounted.path())

        other = Folder(parent=None, name="other", kind="test.root", root_path=tmp_path)
        with pytest.raises(ValueError):
            Folder(parent=other, name="beta", kind="test.child", root_path=tmp_path)

        root = Folder(parent=None, name="root", kind="test.root", root_path=tmp_path)
        mid = Folder(parent=root, name="mid", kind="test.mid")
        leaf = Folder(parent=mid, name="leaf", kind="test.leaf")
        assert Path(leaf.path()) == Path(root.path()) / "mid" / "leaf"

    def test_children_lists_materialized_subfolders_and_filters_by_kind(
        self, tmp_path: Path
    ) -> None:
        """``children()`` lists materialized subfolders and filters by ``kind=``;
        a non-materialized parent returns ``[]`` without a mkdir side effect."""
        fresh = Folder(parent=None, name="fresh", kind="test.root", root_path=tmp_path)
        assert fresh.children() == []
        assert not (tmp_path / "fresh").exists(), "children() must not mkdir"

        parent = Folder(parent=None, name="parent", kind="test.root", root_path=tmp_path)
        for name, kind in (
            ("alpha1", "test.alpha"),
            ("alpha2", "test.alpha"),
            ("beta", "test.beta"),
        ):
            _TestSubFolder(parent=parent, name=name, kind=kind).materialize()

        assert len(parent.children()) == 3
        assert all(isinstance(c, Folder) for c in parent.children())
        assert {c.metadata.name for c in parent.children(kind="test.alpha")} == {"alpha1", "alpha2"}
        assert [c.metadata.name for c in parent.children(kind="test.beta")] == ["beta"]

    def test_save_bumps_updated_at_past_created_at(self, tmp_path: Path) -> None:
        """Regression guard on the deliberate deviation from the mtime-based
        design (sub-spec 01 §6): ``save()`` advances ``updated_at`` monotonically
        inside the metadata JSON, decoupled from filesystem mtime."""
        folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)
        folder.materialize()

        time.sleep(0.001)  # let the clock tick so the bump is observable
        folder.save()

        loaded = _load_metadata(FolderMetadata, Path(folder.path()) / "metadata.json")
        assert isinstance(loaded, FolderMetadata)
        assert loaded.updated_at > loaded.created_at

    def test_delete_removes_directory_tree(self, tmp_path: Path) -> None:
        """``delete()`` removes the directory tree (including nested files)."""
        folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)
        folder.materialize()
        folder.write_json("file.json", {})

        captured = Path(folder.path())  # capture before delete (re-path would re-mkdir)
        assert captured.exists()
        folder.delete()
        assert not captured.exists()

    def test_move_to_relocates_and_bumps_updated_at(self, tmp_path: Path) -> None:
        """``move_to(new_parent)`` relocates on disk, reparents, and bumps
        ``updated_at``."""
        parent_a = Folder(parent=None, name="parent_a", kind="test.root", root_path=tmp_path)
        parent_b = Folder(parent=None, name="parent_b", kind="test.root", root_path=tmp_path)

        folder = Folder(parent=parent_a, name="movable", kind="test.child")
        folder.materialize()
        old_path = Path(folder.path())
        before = folder.metadata.updated_at

        time.sleep(0.001)
        folder.move_to(parent_b)

        assert not old_path.exists()
        assert folder.parent is parent_b
        assert Path(folder.path()) == Path(parent_b.path()) / "movable"
        assert Path(folder.path()).exists()
        assert folder.metadata.updated_at > before

    def test_move_to_collision_raises(self, tmp_path: Path) -> None:
        """``move_to`` raises ``FolderMoveCollisionError`` when the target exists."""
        parent_a = Folder(parent=None, name="parent_a", kind="test.root", root_path=tmp_path)
        parent_b = Folder(parent=None, name="parent_b", kind="test.root", root_path=tmp_path)

        folder = Folder(parent=parent_a, name="movable", kind="test.child")
        folder.materialize()
        Path(parent_b.path()).joinpath("movable").mkdir(parents=True, exist_ok=True)

        with pytest.raises(FolderMoveCollisionError):
            folder.move_to(parent_b)

    @pytest.mark.parametrize("bad_name", ["a/b.json", ".."])
    def test_write_json_rejects_path_separators_and_traversal(
        self, tmp_path: Path, bad_name: str
    ) -> None:
        """``write_json`` rejects names with path separators or ``.``/``..``."""
        folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)
        folder.materialize()
        with pytest.raises(ValueError):
            folder.write_json(bad_name, {})


def _concept_folder(name: str, root: Path) -> Folder:
    """A materialized base Concept dir (metadata.json + meta.yaml) for edge tests."""
    folder = Folder(name=name, kind="bundle.concept", root_path=str(root))
    folder.materialize()
    folder.write_meta()
    return folder


class TestFolderEdges:
    def test_append_link_round_trips_role_and_path_view(self, tmp_path: Path) -> None:
        """``append_link`` writes a typed edge recoverable via ``typed_out_edges``
        (role intact) while the path-only ``out_edges`` view still resolves the
        target."""
        src = _concept_folder("src", tmp_path)
        dst = _concept_folder("dst", tmp_path)
        append_link(src, dst, role="derived_from")

        typed = src.typed_out_edges()
        assert len(typed) == 1
        assert os.path.normpath(typed[0].target) == os.path.normpath(str(dst.resolve()))
        assert typed[0].role == "derived_from"
        assert os.path.normpath(str(dst.resolve())) in {
            os.path.normpath(e) for e in src.out_edges()
        }

    def test_legacy_untyped_link_defaults_role_and_is_kept(self, tmp_path: Path) -> None:
        """A plain pre-role markdown link parses to ``DEFAULT_EDGE_ROLE`` and is
        never dropped; ``meta.yaml`` is never consulted for the edge."""
        from molexp.workspace.edges import DEFAULT_EDGE_ROLE

        src = _concept_folder("src", tmp_path)
        dst = _concept_folder("dst", tmp_path)
        src.write_index("# src\n\n- [dst](../dst)\n")

        typed = src.typed_out_edges()
        assert len(typed) == 1
        assert typed[0].role == DEFAULT_EDGE_ROLE
        assert os.path.normpath(str(dst.resolve())) in {
            os.path.normpath(e) for e in src.out_edges()
        }
        assert "dst" not in (Path(src.resolve()) / "meta.yaml").read_text(encoding="utf-8")

    def test_append_link_unknown_role_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        src = _concept_folder("src", tmp_path)
        dst = _concept_folder("dst", tmp_path)
        with pytest.raises(ValueError):
            append_link(src, dst, role="not_a_role")
        assert src.read_index() == ""


def test_import_guard_folder_pulls_no_upstream_layer() -> None:
    """``import molexp.workspace.folder`` pulls no upstream layer (workflow /
    agent) nor ``pydantic_ai`` / ``pydantic_graph`` into ``sys.modules``.
    Subprocess-isolated because the in-process interpreter has those loaded."""
    code = (
        "import sys\n"
        "import molexp.workspace.folder  # noqa: F401\n"
        "for mod in ('molexp.workflow', 'molexp.agent', 'pydantic_ai', 'pydantic_graph'):\n"
        "    assert mod not in sys.modules, "
        "        f'molexp.workspace.folder eagerly imported {mod}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
    if result.returncode != 0:
        print("stderr:", result.stderr.decode())
        print("stdout:", result.stdout.decode())
    assert result.returncode == 0, "import-guard subprocess failed; see captured stderr above"
