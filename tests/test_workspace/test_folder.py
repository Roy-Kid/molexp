"""Tests for ``molexp.workspace.folder`` — the abstract Folder base class.

This module covers sub-spec ``unify-folder-abstraction-01-folder-base``.
The 11 acceptance criteria (ac-001 .. ac-011) are mapped one-to-one onto
test methods below; helper edge-case tests are kept at the bottom of the
file.

Production code does not exist yet — every test must FAIL at collection
time (``ImportError`` from ``molexp.workspace.folder``) until sub-spec 01
is implemented.

References:
- spec:        ``.claude/specs/unify-folder-abstraction-01-folder-base.md``
- acceptance:  ``.claude/specs/unify-folder-abstraction-01-folder-base.acceptance.md``
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from molexp.workspace.base import _load_metadata
from molexp.workspace.folder import Folder, FolderMoveCollisionError
from molexp.workspace.models import FolderMetadata


# ── Helper subclass used purely to verify children() filtering. ────────────
#
# ``Folder`` has no business subclasses in sub-spec 01; this private
# ``_TestSubFolder`` exists only so ``children()`` has something to
# reconstruct without depending on sub-spec 02's eventual ``Project`` /
# ``Experiment`` / ``Run`` subclasses.
class _TestSubFolder(Folder):
    """Minimal Folder subclass used only by the children() filter test."""


# ── ac-001 ────────────────────────────────────────────────────────────────
def test_construction_is_side_effect_free_first_path_creates_dir(
    tmp_path: Path,
) -> None:
    """Folder construction performs no I/O; ``path()`` is the lazy mkdir.

    Covers ac-001: instantiating ``Folder`` must not touch the filesystem.
    Calling ``path()`` once creates the directory; calling it again is an
    idempotent no-op returning the same Path.
    """
    folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)

    target = tmp_path / "alpha"
    assert not target.exists(), "construction must be side-effect-free"

    first = Path(folder.path())
    assert first == target
    assert first.is_dir()

    second = Path(folder.path())
    assert second == first
    assert second.is_dir()


# ── ac-002 ────────────────────────────────────────────────────────────────
def test_write_json_persists_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``write_json`` happy path round-trips; mid-write failure is atomic.

    Covers ac-002:
    - happy path: ``folder.write_json("data.json", {"k": 1})`` then stdlib
      ``json.load`` returns ``{"k": 1}``.
    - failure path: monkeypatch ``os.replace`` to raise; the destination
      file (or its absence on first write) is unchanged after the failure.
    """
    folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)

    # Happy path.
    written = folder.write_json("data.json", {"k": 1})
    written_local = Path(str(written))
    assert written_local.exists()
    with written_local.open() as fh:
        loaded = json.load(fh)
    assert loaded == {"k": 1}

    # Failure path: original file (here: pre-existing payload) survives a
    # mid-write os.replace failure on the *next* write attempt.
    target_path = Path(folder.path()) / "data.json"

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated mid-write failure")

    # Patch the os.replace symbol in molexp.workspace.base (where
    # atomic_write_json calls it).
    monkeypatch.setattr("molexp.workspace.base.os.replace", _explode)

    with pytest.raises(RuntimeError):
        folder.write_json("data.json", {"k": 2})

    # Original file content is intact.
    with target_path.open() as fh:
        still = json.load(fh)
    assert still == {"k": 1}


# ── ac-003 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, valid",
    [
        ("ok-1", True),
        ("UPPER", False),
        ("../etc", False),
        ("", False),
        (".  ", False),
        ("   ", False),
        ("!!!", False),  # all-punctuation slugifies to empty
    ],
)
def test_id_name_kind_validation_name(tmp_path: Path, name: str, valid: bool) -> None:
    """``name`` is validated and slugified into a kind-pattern-conforming id.

    Covers ac-003 (name axis). ``slugify(name)`` must yield an ``id`` that
    matches ``_KIND_PATTERN``; otherwise ``ValueError`` at construction.
    All-punctuation names that slugify to empty are rejected.
    """
    if valid:
        Folder(parent=None, name=name, kind="test.root", root_path=tmp_path)
    else:
        with pytest.raises(ValueError):
            Folder(parent=None, name=name, kind="test.root", root_path=tmp_path)


@pytest.mark.parametrize(
    "kind, valid",
    [
        ("workspace.project", True),
        ("test.root", True),
        ("agent", True),
        ("WORKSPACE.foo", False),
        (".leading", False),
        ("../etc", False),
        ("", False),
        ("   ", False),
        ("foo bar", False),
    ],
)
def test_id_name_kind_validation_kind(tmp_path: Path, kind: str, valid: bool) -> None:
    """``kind`` is validated against ``_KIND_PATTERN`` at construction.

    Covers ac-003 (kind axis). dotted lowercase ASCII only; uppercase,
    leading-dot, and path-traversal kinds are rejected.
    """
    if valid:
        Folder(parent=None, name="alpha", kind=kind, root_path=tmp_path)
    else:
        with pytest.raises(ValueError):
            Folder(parent=None, name="alpha", kind=kind, root_path=tmp_path)


# ── ac-004 ────────────────────────────────────────────────────────────────
def test_parent_root_path_mutual_exclusion(tmp_path: Path) -> None:
    """Construction shapes after the sub-spec 03 CRUD rewrite.

    - ``parent=None`` with ``root_path=None`` is the **unmounted state** —
      legal at construction time but ``.path()`` and ``.materialize()``
      raise ``RuntimeError`` until ``parent.add_folder(child)`` mounts it.
    - ``parent=other`` with ``root_path`` set → ``ValueError`` (still
      mutually exclusive when both are non-None).
    - 3-level nesting walks correctly: ``Path(leaf.path()) == root / mid / leaf``.
    """
    # parent=None with root_path=None is now ALLOWED (unmounted state).
    unmounted = Folder(parent=None, name="alpha", kind="test.root", root_path=None)
    assert unmounted._parent is None
    with pytest.raises(RuntimeError, match="unmounted"):
        Path(unmounted.path())

    other = Folder(parent=None, name="other", kind="test.root", root_path=tmp_path)

    # parent + root_path both given → still ValueError.
    with pytest.raises(ValueError):
        Folder(parent=other, name="beta", kind="test.child", root_path=tmp_path)

    # Three-level nesting walks correctly.
    root = Folder(parent=None, name="root", kind="test.root", root_path=tmp_path)
    mid = Folder(parent=root, name="mid", kind="test.mid")
    leaf = Folder(parent=mid, name="leaf", kind="test.leaf")
    assert Path(leaf.path()) == Path(root.path()) / "mid" / "leaf"


# ── ac-005 ────────────────────────────────────────────────────────────────
def test_children_lists_and_filters_by_kind(tmp_path: Path) -> None:
    """``children()`` lists materialized subfolders; ``kind=`` filters.

    Covers ac-005:
    - Two ``test.alpha`` + one ``test.beta`` materialized under a parent.
    - ``parent.children()`` returns 3 Folder instances.
    - ``parent.children(kind="test.alpha")`` returns exactly 2.
    - ``parent.children()`` on a non-materialized parent returns ``[]``
      (lazy semantics — no exception, no mkdir side effect).
    """
    parent = Folder(parent=None, name="parent", kind="test.root", root_path=tmp_path)

    # Lazy semantics: pre-materialize, children() returns [].
    # Use a fresh non-materialized folder so we can test the empty case
    # without forcing Path(parent.path()) to mkdir.
    fresh = Folder(parent=None, name="fresh", kind="test.root", root_path=tmp_path)
    assert fresh.children() == []
    assert not (tmp_path / "fresh").exists(), "children() must not mkdir"

    # Materialize three sub-folders under parent.
    alpha1 = _TestSubFolder(parent=parent, name="alpha1", kind="test.alpha")
    alpha2 = _TestSubFolder(parent=parent, name="alpha2", kind="test.alpha")
    beta = _TestSubFolder(parent=parent, name="beta", kind="test.beta")
    alpha1.materialize()
    alpha2.materialize()
    beta.materialize()

    all_children = parent.children()
    assert len(all_children) == 3
    assert all(isinstance(c, Folder) for c in all_children)

    alphas = parent.children(kind="test.alpha")
    assert len(alphas) == 2
    assert {c.metadata.name for c in alphas} == {"alpha1", "alpha2"}

    betas = parent.children(kind="test.beta")
    assert len(betas) == 1
    assert betas[0].metadata.name == "beta"


# ── ac-006 ────────────────────────────────────────────────────────────────
def test_folder_metadata_round_trip_with_monotonic_updated_at(
    tmp_path: Path,
) -> None:
    """FolderMetadata round-trips with ``updated_at > created_at``.

    Covers ac-006. This test deliberately deviates from
    ``models.py:38``'s mtime-based design (the ``WorkspaceMetadata`` /
    ``ProjectMetadata`` / etc. comment that says "No ``updated_at``. If
    you need 'last modified', read the file mtime."). Rationale (from
    sub-spec 01 § Design § 6):

    - ``mtime`` is unstable across rsync / git checkout / cross-host
      copies, and cannot back the global folder index introduced in
      sub-spec 03;
    - the lifecycle ``updated_at`` must be monotonically advanced inside
      the metadata JSON, decoupled from external filesystem state.

    This test pins that behaviour as a regression guard so the deviation
    is not silently reverted.
    """
    folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)
    folder.materialize()

    # Monotonicity guard: ensure the clock has had a chance to tick.
    time.sleep(0.001)
    folder.save()

    loaded = _load_metadata(FolderMetadata, Path(folder.path()) / "metadata.json")
    assert isinstance(loaded, FolderMetadata)
    assert loaded.updated_at > loaded.created_at, (
        "save() must bump updated_at past created_at — this is the "
        "deliberate deviation from models.py:38's mtime-based design."
    )


# ── ac-007 ────────────────────────────────────────────────────────────────
def test_delete_removes_directory_recursively(tmp_path: Path) -> None:
    """``delete()`` removes the directory tree (including nested files).

    Covers ac-007. We capture ``path()`` BEFORE ``delete()`` because
    re-calling ``Path(folder.path())`` afterward would re-mkdir the directory
    (lazy mkdir contract from ac-001).
    """
    folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)
    folder.materialize()
    folder.write_json("file.json", {})

    captured = Path(folder.path())
    assert captured.exists()

    folder.delete()

    # Do NOT call Path(folder.path()) here; it would re-mkdir.
    assert not captured.exists()


# ── ac-008 ────────────────────────────────────────────────────────────────
def test_move_to_relocates_and_bumps_updated_at(tmp_path: Path) -> None:
    """``move_to(new_parent)`` relocates on disk and bumps ``updated_at``.

    Covers ac-008 (success path). Old path absent, new path present,
    ``folder.parent is parent_b``, ``metadata.updated_at`` advanced past
    its pre-move value.
    """
    parent_a = Folder(parent=None, name="parent_a", kind="test.root", root_path=tmp_path)
    parent_b = Folder(parent=None, name="parent_b", kind="test.root", root_path=tmp_path)

    folder = Folder(parent=parent_a, name="movable", kind="test.child")
    folder.materialize()

    old_path = Path(folder.path())
    before = folder.metadata.updated_at
    assert old_path.exists()

    time.sleep(0.001)
    folder.move_to(parent_b)

    assert not old_path.exists(), "old path must be gone after move"
    assert folder.parent is parent_b
    new_path = Path(folder.path())
    assert new_path == Path(parent_b.path()) / "movable"
    assert new_path.exists()
    assert folder.metadata.updated_at > before


def test_move_to_collision_raises(tmp_path: Path) -> None:
    """``move_to`` raises ``FolderMoveCollisionError`` on existing target.

    Covers ac-008 (collision path).
    """
    parent_a = Folder(parent=None, name="parent_a", kind="test.root", root_path=tmp_path)
    parent_b = Folder(parent=None, name="parent_b", kind="test.root", root_path=tmp_path)

    folder = Folder(parent=parent_a, name="movable", kind="test.child")
    folder.materialize()

    # Pre-create a colliding directory at the target.
    Path(parent_b.path()).joinpath("movable").mkdir(parents=True, exist_ok=True)

    with pytest.raises(FolderMoveCollisionError):
        folder.move_to(parent_b)


# ── ac-009: legacy reverse-import (deleted by unify-folder-abstraction-03) ──
# ``subsystem.py`` was removed; ``_KIND_PATTERN`` now lives only in folder.py.


# ── ac-010 ────────────────────────────────────────────────────────────────
def test_public_exports_resolve() -> None:
    """``Folder``, ``FolderMetadata``, ``FolderMoveCollisionError`` are public.

    Covers ac-010.
    """
    from molexp.workspace import (
        Folder as PublicFolder,
    )
    from molexp.workspace import (
        FolderMetadata as PublicFolderMetadata,
    )
    from molexp.workspace import (
        FolderMoveCollisionError as PublicFolderMoveCollisionError,
    )

    assert PublicFolder is Folder
    assert PublicFolderMetadata is FolderMetadata
    assert PublicFolderMoveCollisionError is FolderMoveCollisionError

    import molexp.workspace as ws

    assert "Folder" in ws.__all__
    assert "FolderMetadata" in ws.__all__
    assert "FolderMoveCollisionError" in ws.__all__


# ── ac-011 ────────────────────────────────────────────────────────────────
def test_import_guard_subprocess() -> None:
    """``import molexp.workspace.folder`` pulls no upstream layer modules.

    Covers ac-011. Subprocess isolation is required because the in-process
    interpreter likely has ``molexp.workflow`` / ``molexp.agent`` etc.
    already loaded from other test modules.
    """
    code = (
        "import sys\n"
        "import molexp.workspace.folder  # noqa: F401\n"
        "assert 'molexp.workflow' not in sys.modules, "
        "    'molexp.workspace.folder eagerly imported molexp.workflow'\n"
        "assert 'molexp.agent' not in sys.modules, "
        "    'molexp.workspace.folder eagerly imported molexp.agent'\n"
        "assert 'pydantic_ai' not in sys.modules, "
        "    'molexp.workspace.folder eagerly imported pydantic_ai'\n"
        "assert 'pydantic_graph' not in sys.modules, "
        "    'molexp.workspace.folder eagerly imported pydantic_graph'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print("stderr:", result.stderr.decode())
        print("stdout:", result.stdout.decode())
    assert result.returncode == 0, "import-guard subprocess failed; see captured stderr above"


# ── Edge cases (extra, not strictly tied to ac) ────────────────────────────
def test_read_json_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    """``read_json`` on a missing file surfaces stdlib ``FileNotFoundError``.

    Per spec § Testing strategy: transparent stdlib propagation, not a
    custom exception wrapper.
    """
    folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)
    folder.materialize()
    with pytest.raises(FileNotFoundError):
        folder.read_json("does-not-exist.json")


@pytest.mark.parametrize(
    "bad_name",
    [
        "a/b.json",
        "a\\b.json",
        ".",
        "..",
    ],
)
def test_write_json_rejects_bad_names(tmp_path: Path, bad_name: str) -> None:
    """``write_json`` rejects names with path separators or ``.``/``..``.

    Mirrors ``SubsystemStore.file`` validation semantics (subsystem.py:81).
    """
    folder = Folder(parent=None, name="alpha", kind="test.root", root_path=tmp_path)
    folder.materialize()
    with pytest.raises(ValueError):
        folder.write_json(bad_name, {})
