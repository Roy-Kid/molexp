"""Unified folder abstraction for workspace storage.

Introduces ``Folder``: a plain Python class providing the contract every
directory under a workspace satisfies — lazy mkdir, atomic JSON, id /
name / kind validation, parent pointer, generic five-verb CRUD
(``add_folder`` / ``get_folder`` / ``has_folder`` / ``list_folders`` /
``remove_folder``), one plural JSON children-index per class, lifecycle
metadata, and delete / move operations.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path as _StdPath
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar, NamedTuple, TypeVar, cast

from molexp._typing import JSONValue
from molexp.atomicio import file_lock
from molexp.knowledge.types import resolve_concept_type
from molexp.path import Path

from .base import _reconstruct
from .edges import DEFAULT_EDGE_ROLE, Edge, EdgeRole, encode_label, parse_role, validate_role
from .errors import FolderMoveCollisionError
from .fs import FileSystem, PathArg
from .fs_local import LocalFileSystem
from .models import FolderMetadata
from .utils import slugify

if TYPE_CHECKING:
    from .file_store import FileStore

F = TypeVar("F", bound="Folder")

WORKSPACE_ROOT_KIND = "workspace.root"
WORKSPACE_PROJECT_KIND = "workspace.project"
WORKSPACE_EXPERIMENT_KIND = "workspace.experiment"
WORKSPACE_RUN_KIND = "workspace.run"

_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)*$")

_FORBIDDEN_FILE_NAMES = {".", ".."}
_CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")

# ── OKF: narrative index.md + JSON concept identity ──────────────────────────
# One structured format on disk: **JSON**. Every Concept dir carries
# ``meta.json`` (``type`` → registry; path basename = id). WPER domain records
# use class-named entity JSON (``project.json`` / ``run.json`` / …). Narrative
# stays in ``index.md`` (Markdown is not a data format for structured fields).
INDEX_FILENAME = "index.md"
META_JSON_FILENAME = "meta.json"  # sole concept identity file (type → registry)
OPS_DIR = "ops"  # operational sidecar — hot machine state, not knowledge
LEGACY_OPS_DIR = "_ops"  # read fallback for workspaces written before the rename
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")  # [label](target) — both captured


def _parse_iso_datetime(raw: object, *, default: datetime) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return default
    return default


def _folder_metadata_from_marker(
    child_dir: PathArg,
    marker: dict[str, object],
) -> FolderMetadata:
    """Build in-memory :class:`FolderMetadata` from ``meta.json`` + dir name.

    Path basename is identity — marker ``id`` is ignored when present (legacy).
    """
    slug = PurePosixPath(str(child_dir)).name
    kind = str(marker.get("type") or "").strip() or "concept"
    now = datetime.now(UTC)
    created = _parse_iso_datetime(marker.get("created_at"), default=now)
    updated = _parse_iso_datetime(marker.get("updated_at"), default=created)
    extra_raw = marker.get("extra")
    extra: dict[str, JSONValue] = (
        cast("dict[str, JSONValue]", extra_raw) if isinstance(extra_raw, dict) else {}
    )
    return FolderMetadata(
        id=slug,
        name=slug,
        kind=kind,
        created_at=created,
        updated_at=updated,
        extra=extra,
    )


def _load_concept_marker_dict(fs: FileSystem, concept_dir: PathArg) -> dict[str, object] | None:
    """Load concept identity dict from ``meta.json``."""
    primary = fs.join(concept_dir, META_JSON_FILENAME)
    if not fs.exists(primary):
        return None
    with fs.open(primary) as fh:
        raw: object = json.load(fh)
    return cast("dict[str, object]", raw) if isinstance(raw, dict) else None


class LinkScan(NamedTuple):
    """Resolved out-links of a Folder's ``index.md``.

    Attributes:
        concepts: Targets resolving to an existing in-tree dir — the
            knowledge-graph out-edges (path-only, unchanged for back-compat).
        external: ``http(s)://`` links.
        other: In-tree targets that don't resolve to a dir.
        typed_concepts: The same in-tree concept edges as :attr:`concepts`, each
            paired with its declared :class:`~molexp.workspace.edges.EdgeRole`
            recovered from the markdown ``[label]`` channel.
    """

    concepts: list[str]
    external: list[str]
    other: list[str]
    typed_concepts: list[Edge]


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise ValueError("folder kind must be a non-empty string")
    if not _KIND_PATTERN.fullmatch(kind):
        raise ValueError(
            f"invalid folder kind {kind!r}: must be dotted lowercase ASCII "
            "(e.g. 'workspace.project'); no path separators, leading dots, "
            "uppercase, or whitespace"
        )


def _validate_file_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("folder file name must be a non-empty string")
    if name in _FORBIDDEN_FILE_NAMES or "/" in name or "\\" in name:
        raise ValueError(f"invalid folder file name {name!r}")


def _validate_name_to_id(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("folder name must be a non-empty string")
    if not _KIND_PATTERN.fullmatch(name):
        raise ValueError(
            f"invalid folder name {name!r}: must be dotted lowercase ASCII "
            "(e.g. 'my-project'); no path separators, leading dots, "
            "uppercase, or whitespace"
        )
    derived = slugify(name)
    if not derived or not _KIND_PATTERN.fullmatch(derived):
        raise ValueError(f"folder name {name!r} produced invalid id {derived!r}")
    return derived


def _validate_target_registered(workspace: object, target: str | None) -> None:
    """Reject a *target* that is not in the workspace's compute-target registry.

    No-op when *target* is ``None`` or the registry is empty (a registry-less
    workspace keeps accepting free-form target strings — back-compat). Once a
    workspace registers any target, references must name a registered one
    (models.py: ``RunMetadata.target`` is "validated against
    WorkspaceMetadata.targets at write time").
    """
    if target is None:
        return
    metadata = getattr(workspace, "metadata", None)
    registered = getattr(metadata, "targets", ()) or ()
    if registered and not any(getattr(t, "name", None) == target for t in registered):
        names = sorted(getattr(t, "name", "?") for t in registered)
        raise ValueError(
            f"unknown compute target {target!r}: not in the workspace target "
            f"registry {names}; register it first (e.g. `molexp target add`)."
        )


class Folder:
    """A location on a workspace disk.

    The disk itself is :attr:`Workspace.fs` (a :class:`FileSystem`: local,
    remote, or cached). A Folder does not own a disk — it resolves I/O
    through the workspace it is attached to. User byte writes go through
    :attr:`files` (a :class:`FileStore` rooted here).

    Carries a ``parent`` pointer, lazy materialization, atomic JSON IO,
    children listing, lifecycle metadata, delete / move operations, and
    the generic ``attach`` / ``create_child`` / ``get_child`` triplet.
    """

    _exists_error_cls: ClassVar[type[Exception]] = ValueError
    _not_found_error_cls: ClassVar[type[Exception]] = LookupError

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str,
        root_path: PathArg | None = None,
        fs: FileSystem | None = None,
    ) -> None:
        if parent is not None and root_path is not None:
            raise ValueError(
                "Folder: parent and root_path are mutually exclusive — "
                "non-root folders inherit their parent's path"
            )
        _validate_kind(kind)
        derived_id = _validate_name_to_id(name)
        self._parent = parent
        self._name = derived_id
        self._kind = kind
        self._root_path: Path | None = Path(os.fspath(root_path)) if root_path is not None else None
        if fs is not None:
            self._disk_backend = fs
        elif parent is None:
            self._disk_backend = LocalFileSystem()
        else:
            self._disk_backend = None
        self._metadata = FolderMetadata(id=derived_id, name=name, kind=kind)
        self._children_cache: dict[str, Folder] = {}

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def parent(self) -> Folder | None:
        return self._parent

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def metadata(self) -> FolderMetadata:
        return self._metadata

    @property
    def folder_metadata(self) -> FolderMetadata:
        return self._metadata

    def _disk(self) -> FileSystem:
        """The disk this location lives on (walks to the workspace root)."""
        node: Folder | None = self
        while node is not None:
            backend = getattr(node, "_disk_backend", None)
            if backend is not None:
                return backend
            node = node._parent
        return LocalFileSystem()

    @property
    def files(self) -> FileStore:
        """Byte-exit rooted at this folder. Does not own the disk."""
        from .file_store import FileStore

        return FileStore(self.path(), fs=self._disk())

    # ── Path resolution ──────────────────────────────────────────────────
    #
    # ``resolve`` and ``path`` return :class:`molexp.Path` — a subclass of
    # :class:`pathlib.PurePosixPath`.  It is a *pure* path: no ``.exists``,
    # ``.read_text`` or other I/O methods, so it cannot accidentally
    # short-circuit to the local filesystem on a remote-backed folder.
    # All I/O flows through :meth:`_disk` (the workspace FileSystem).

    def path(self) -> Path:
        """Return the on-disk path; create only if missing (lazy, idempotent).

        Pure path math is :meth:`resolve`.  This twin only ``mkdir`` when
        the directory is not already present — so remote-backed folders that
        already exist never issue a write (remote ``mkdir`` can fail on
        permission even when the path is already a directory).
        """
        target = self.resolve()
        if self._disk().is_dir(target):
            return target
        self._disk().mkdir(target, parents=True, exist_ok=True)
        return target

    def resolve(self) -> Path:
        """Walk the parent chain without triggering lazy mkdir."""
        if self._parent is None:
            if self._root_path is None:
                raise RuntimeError(
                    f"folder {self._name!r} (kind={self._kind!r}) is unmounted — "
                    "construct via parent.add_folder(child) or pass root_path="
                )
            return Path(self._disk().join(self._root_path, self._name))
        return Path(self._disk().join(self._parent.resolve(), self._name))

    # ── Index filename ───────────────────────────────────────────────────
    #
    # Children-index lives on the PARENT and lists children of *this* class.
    # One file, JSON only: the **plural** of the child kind
    # (``experiments.json``, ``runs.json``, ``projects.json``) so it never
    # collides with the **singular** entity file on the child dir
    # (``experiment.json``, ``run.json``, ``project.json``).

    @classmethod
    def _index_filename(cls) -> str:
        """Canonical children-index basename (plural of the child kind)."""
        singular = _CAMEL_TO_SNAKE.sub("_", cls.__name__).lower()
        return f"{singular}s.json"

    def _index_path_for(self, cls: type[Folder]) -> str:
        """Resolved path of the children-index for *cls* under this folder."""
        return self._disk().join(self.resolve(), cls._index_filename())

    # ── Atomic JSON IO ───────────────────────────────────────────────────

    def read_json(self, name: str) -> dict[str, JSONValue]:
        _validate_file_name(name)
        fpath = self._disk().join(self.path(), name)
        with self._disk().open(fpath) as fh:
            raw: object = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{fpath} top-level JSON must be an object, got {type(raw).__name__}")
        return cast("dict[str, JSONValue]", raw)

    def write_json(self, name: str, data: object) -> str:
        _validate_file_name(name)
        fpath = self._disk().join(self.path(), name)
        self._disk().atomic_write_json(fpath, data)
        return fpath

    # ── OKF meta.json (sole concept identity; type → knowledge registry) ──

    def write_meta(self) -> str:
        """Write the OKF ``meta.json`` concept identity (``type`` = this kind).

        One structured format on disk: JSON (same family as entity ``*.json``).

        * required ``type`` — registered concept kind (path basename = id)
        * optional ``created_at`` / ``updated_at`` — lifecycle when there is no
          class-named entity JSON (generic :class:`Folder`)
        * optional ``extra`` — free-form map

        Domain entities keep class-named JSON for business fields. Subclasses
        with a rich typed head (Note, Agent, …) override this method.
        """
        data: dict[str, JSONValue] = {
            "type": self._kind,
            "created_at": self._metadata.created_at.isoformat(),
            "updated_at": self._metadata.updated_at.isoformat(),
        }
        if self._metadata.extra:
            data["extra"] = self._metadata.extra
        fpath = self._disk().join(self.path(), META_JSON_FILENAME)
        self._disk().atomic_write_json(fpath, data)
        return fpath

    def read_meta(self) -> dict[str, JSONValue]:
        """Read the OKF ``meta.json`` marker, or ``{}`` if absent."""
        raw = _load_concept_marker_dict(self._disk(), self.resolve())
        return cast("dict[str, JSONValue]", raw) if raw is not None else {}

    # ── OKF narrative + markdown-link knowledge graph ─────────────────────

    def read_index(self) -> str:
        """Return the OKF ``index.md`` narrative, or ``""`` if absent."""
        fpath = self._disk().join(self.resolve(), INDEX_FILENAME)
        return self._disk().read_text(fpath) if self._disk().exists(fpath) else ""

    def write_index(self, text: str) -> str:
        """Atomically write the OKF ``index.md`` narrative + markdown links."""
        fpath = self._disk().join(self.path(), INDEX_FILENAME)
        self._disk().atomic_write_text(fpath, text)
        return fpath

    def links(self) -> LinkScan:
        """Parse ``index.md`` markdown links, classified (see :class:`LinkScan`).

        Targets resolve relative to this Folder's dir; a trailing ``index.md``
        is stripped to its containing dir. An in-tree target counts as a
        knowledge-graph edge when it resolves to an existing dir.
        """
        base = PurePosixPath(str(self.resolve()))
        concepts: list[str] = []
        external: list[str] = []
        other: list[str] = []
        typed_concepts: list[Edge] = []
        for raw_label, target in _MD_LINK.findall(self.read_index()):
            if target.startswith(("http://", "https://")):
                external.append(target)
                continue
            norm = PurePosixPath(os.path.normpath(base / target))
            concept_dir = norm.parent if norm.name == INDEX_FILENAME else norm
            if self._disk().is_dir(str(concept_dir)):
                concepts.append(str(concept_dir))
                role, _human = parse_role(raw_label)
                typed_concepts.append(Edge(target=str(concept_dir), role=role))
            else:
                other.append(target)
        return LinkScan(
            concepts=concepts,
            external=external,
            other=other,
            typed_concepts=typed_concepts,
        )

    def out_edges(self) -> list[str]:
        """In-tree Folder link targets — the knowledge-graph out-edges (path-only)."""
        return self.links().concepts

    def typed_out_edges(self) -> list[Edge]:
        """In-tree out-edges paired with their declared ``EdgeRole``.

        The typed companion to :meth:`out_edges`: each :class:`~molexp.workspace.edges.Edge`
        carries the same resolved target path plus the role recovered from the
        markdown label channel (a legacy untyped link defaults to
        :data:`~molexp.workspace.edges.DEFAULT_EDGE_ROLE`, never dropped).
        """
        return self.links().typed_concepts

    # ── ops/ operational sidecar (hot machine state, never in meta.json) ─

    def _ops_json_path(self, name: str) -> str:
        """Return the existing ops JSON path (``ops/`` first, then ``_ops/``).

        Writes always go to :data:`OPS_DIR`. Reads fall back to the pre-rename
        ``_ops/`` location so existing workspaces still load.
        """
        canonical = self._disk().join(self.resolve(), OPS_DIR, f"{name}.json")
        if self._disk().exists(canonical):
            return canonical
        legacy = self._disk().join(self.resolve(), LEGACY_OPS_DIR, f"{name}.json")
        if self._disk().exists(legacy):
            return legacy
        return canonical

    def ops_dir(self) -> str:
        """Return the per-Folder ``ops/`` sidecar dir, creating it if absent."""
        d = self._disk().join(self.path(), OPS_DIR)
        self._disk().mkdir(d, parents=True, exist_ok=True)
        return d

    def read_ops_json(self, name: str) -> dict[str, JSONValue] | None:
        """Read ``ops/<name>.json`` (or legacy ``_ops/<name>.json``)."""
        fpath = self._ops_json_path(name)
        if not self._disk().exists(fpath):
            return None
        with self._disk().open(fpath) as fh:
            return cast("dict[str, JSONValue]", json.load(fh))

    def write_ops_json(self, name: str, data: object) -> None:
        """Atomically write ``ops/<name>.json`` (operational state)."""
        if isinstance(data, (dict, list, str, bytes)):
            self.files.put(f"{OPS_DIR}/{name}.json", data)
        else:
            self._disk().atomic_write_json(self._disk().join(self.ops_dir(), f"{name}.json"), data)

    def update_ops_json(
        self, name: str, fn: Callable[[dict[str, JSONValue]], dict[str, JSONValue]]
    ) -> dict[str, JSONValue]:
        """Read-modify-write ``ops/<name>.json`` under an advisory file lock.

        The lock is a local file lock (``molexp.atomicio.file_lock``); concurrent
        same-host RMW is safe. (Remote-backend locking is a future refinement.)
        """
        ops = self.ops_dir()
        with file_lock(_StdPath(self._disk().join(ops, f"{name}.json.lock"))):
            current = self.read_ops_json(name) or {}
            updated = fn(current)
            self.files.put(f"{OPS_DIR}/{name}.json", updated)
        return updated

    # ── Lifecycle ────────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Persist concept identity as ``meta.json`` (creating the dir lazily)."""
        self._metadata = self._metadata.model_copy(update={"updated_at": datetime.now()})
        self.write_meta()

    def save(self) -> None:
        """Bump ``updated_at`` and rewrite ``meta.json``."""
        self._metadata = self._metadata.model_copy(update={"updated_at": datetime.now()})
        self.write_meta()

    # ── Children ─────────────────────────────────────────────────────────

    def _try_child_folder_metadata(self, entry_path: PathArg) -> FolderMetadata | None:
        """Load in-memory FolderMetadata from ``meta.json``."""
        raw = _load_concept_marker_dict(self._disk(), entry_path)
        if raw is None:
            return None
        return _folder_metadata_from_marker(entry_path, raw)

    def children(self, kind: str | None = None) -> list[Folder]:
        self_path = self.resolve()
        if not self._disk().is_dir(self_path):
            return []
        result: list[Folder] = []
        for entry_name in sorted(self._disk().listdir(self_path)):
            if entry_name in _FORBIDDEN_FILE_NAMES:
                continue
            entry_path = self._disk().join(self_path, entry_name)
            if not self._disk().is_dir(entry_path):
                continue
            child_meta = self._try_child_folder_metadata(entry_path)
            if child_meta is None:
                continue
            if kind is not None and child_meta.kind != kind:
                continue
            child = _reconstruct(
                Folder,
                {
                    "_parent": self,
                    "_name": child_meta.id,
                    "_kind": child_meta.kind,
                    "_root_path": None,
                    "_metadata": child_meta,
                    "_children_cache": {},
                },
            )
            result.append(child)
        return result

    # ── Subclass hook contract ────────────────────────────────────────────
    #
    # ``resolve``, ``child_dir`` and ``from_disk`` are **framework hooks** —
    # public protocol surface that the Folder CRUD machinery (``add_folder``,
    # ``get_folder``, ``has_folder``, …) reaches into across class boundaries
    # to wire children to disk. Subclasses override them.
    #
    # Invariants every override MUST hold (failure modes have bitten us in
    # the past, see git history for ``fs``-drop bugs at every level):
    #
    #   * Never store a FileSystem on the child — the disk lives on
    #     Workspace and :meth:`_disk` walks to it. Use
    #     :meth:`base_from_disk_attrs` instead of hand-rolling the attrs
    #     dict — it bakes the invariants in.
    #   * Carry ``FolderMetadata`` (kind, id, *human* name, timestamps)
    #     into ``_metadata`` — don't substitute id for name.
    #   * ``resolve()`` is the side-effect-free path computation; ``path()``
    #     is its mkdir-ing twin. Don't trigger I/O in ``resolve``.
    #   * ``child_dir`` returns the absolute path of a child with this
    #     class's *layout* (e.g. ``projects/<id>``, ``experiments/<id>``).
    #
    # **Return type**: every Folder subclass returns :class:`molexp.Path`
    # (a :class:`pathlib.PurePosixPath` subclass).  Pure POSIX path
    # arithmetic is available (``/`` operator, ``.parent``, ``.name``);
    # I/O methods are deliberately absent so a remote-backed folder cannot
    # silently short-circuit to the local filesystem.  All I/O routes
    # through ``self._disk()``.  This unifies the workspace layer (was: ``str``)
    # and the agent layer (was: ``pathlib.Path``, local-only) — agent
    # subclasses now inherit remote-compat for free.
    #
    # Subclasses with their own entity metadata (Project / Experiment / Run)
    # override ``from_disk`` to load their entity model from a different
    # filename, but must call :meth:`base_from_disk_attrs` to seed the
    # common keys.
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Where a child with *derived_id* lives under *parent*. Override per subclass layout.

        Must use :meth:`resolve` (not :meth:`path`) — this is pure path math
        and must not trigger lazy mkdir on the parent (remote workspaces would
        otherwise attempt writes during list/get).
        """
        return Path(parent._disk().join(parent.resolve(), derived_id))

    @classmethod
    def base_from_disk_attrs(
        cls,
        parent: Folder,
        meta: FolderMetadata,
    ) -> dict[str, object]:
        """Common attrs dict for ``_reconstruct`` — call this from every
        subclass ``from_disk`` to guarantee the parent link (and thus the disk)."""
        return {
            "_parent": parent,
            "_name": meta.id,
            "_kind": meta.kind,
            "_root_path": None,
            "_metadata": meta,
            "_children_cache": {},
        }

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Folder:
        """Generic loader: reconstruct *child_dir* from ``meta.json``.

        Concept identity is ``meta.json`` (``type`` → kind; dir name → id).
        A dir with no marker is not a Folder: ``FileNotFoundError``.
        """
        fs = parent._disk()
        raw = _load_concept_marker_dict(fs, child_dir)
        if raw is None:
            raise FileNotFoundError(fs.join(child_dir, META_JSON_FILENAME))
        child_meta = _folder_metadata_from_marker(child_dir, raw)
        return _reconstruct(cls, cls.base_from_disk_attrs(parent, child_meta))

    # ── Generic five-verb CRUD ───────────────────────────────────────────

    def _construct_child(self, cls: type[F], name: str, **kwargs: object) -> F:
        """Build a typed child folder parented at ``self`` (not yet on disk).

        The single construction hook the typed ``add_*`` sugar
        (:meth:`Workspace.add_project`, :meth:`Project.add_experiment`,
        :meth:`Experiment.add_run`) uses before handing the child to
        :meth:`add_folder`. Entity constructors require a parent, so the child
        is built self-parented; :meth:`add_folder` accepts a self-parented
        child and performs the idempotent mount (cache / on-disk hit / create).
        """
        # Heterogeneous entity constructors (Run/Experiment/Project) all accept
        # ``parent`` + ``name`` plus their own typed kwargs; the dynamic forward
        # is sound at the call sites but not statically checkable here.
        return cls(parent=self, name=name, **kwargs)  # ty: ignore[invalid-argument-type]

    def add_folder(self, child: F) -> F:
        # Accept an unmounted child, or one already parented at ``self`` (the
        # typed ``add_*`` sugar builds self-parented children via
        # ``_construct_child``). Reject a child mounted elsewhere or a root.
        if child._root_path is not None or (
            child._parent is not None and child._parent is not self
        ):
            raise ValueError(
                f"folder {child._name!r} (kind={child._kind!r}) is already mounted; "
                "add_folder() accepts only unmounted or self-parented folders"
            )
        target_cls = type(child)
        slug = child._name
        cached = self._children_cache.get(slug)
        if cached is not None and cached._kind == child._kind:
            # Cache/disk hits reconstruct through type(child), so the narrow
            # return type is truthful (add_folder is generic like get_folder).
            return cast("F", cached)
        child_dir = target_cls.child_dir(self, slug)
        if self._disk().is_dir(child_dir):
            existing = target_cls.from_disk(child_dir, self)
            self._children_cache[slug] = existing
            return cast("F", existing)
        child._parent = self
        child._root_path = None
        if getattr(child, "_disk_backend", None) is not None:
            del child._disk_backend
        child.materialize()  # base: meta.json; WPER subclasses: entity json + write_meta
        if _load_concept_marker_dict(self._disk(), child.resolve()) is None:
            child.write_meta()
        self._children_cache[slug] = child
        self._upsert_index_row(child)
        return child

    def get_folder(self, name: str, *, cls: type[F]) -> F:
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            cached = self._children_cache.get(candidate)
            if isinstance(cached, cls):
                return cached
            child_dir = cls.child_dir(self, candidate)
            if self._disk().is_dir(child_dir):
                loaded = cls.from_disk(child_dir, self)
                if isinstance(loaded, cls):
                    self._children_cache[loaded._name] = loaded
                    return loaded
        raise cls._not_found_error_cls(name)

    def has_folder(self, name: str, *, cls: type[Folder]) -> bool:
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            cached = self._children_cache.get(candidate)
            if isinstance(cached, cls):
                return True
            if cls.child_dir(self, candidate) and self._disk().is_dir(
                cls.child_dir(self, candidate)
            ):
                return True
        return False

    def list_folders(self, *, cls: type[F] | None = None) -> list[F]:
        self_path = self.resolve()
        if not self._disk().is_dir(self_path):
            return []
        out: list[F] = []
        if cls is None:
            for entry_name in sorted(self._disk().listdir(self_path)):
                if entry_name in _FORBIDDEN_FILE_NAMES:
                    continue
                entry_path = self._disk().join(self_path, entry_name)
                if not self._disk().is_dir(entry_path):
                    continue
                child_meta = self._try_child_folder_metadata(entry_path)
                if child_meta is None:
                    continue
                child = _reconstruct(
                    Folder,
                    {
                        "_parent": self,
                        "_name": child_meta.id,
                        "_kind": child_meta.kind,
                        "_root_path": None,
                        "_metadata": child_meta,
                        "_children_cache": {},
                    },
                )
                out.append(cast(F, child))
            return out
        index_path = self._index_path_for(cls)
        if not self._disk().exists(index_path):
            self.sync_folders(cls=cls)
            index_path = self._index_path_for(cls)
            if not self._disk().exists(index_path):
                return []
        try:
            with self._disk().open(index_path) as fh:
                raw: object = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, dict):
            return []
        for slug in raw:
            cached = self._children_cache.get(str(slug))
            if isinstance(cached, cls):
                out.append(cached)
                continue
            child_dir = cls.child_dir(self, str(slug))
            if not self._disk().is_dir(child_dir):
                continue
            try:
                loaded = cls.from_disk(child_dir, self)
            except (FileNotFoundError, OSError):
                continue
            if isinstance(loaded, cls):
                self._children_cache[loaded._name] = loaded
                out.append(loaded)
        return out

    def sync_folders(self, *, cls: type[Folder]) -> None:
        container = cls._container_dir(self)
        index_path = self._index_path_for(cls)
        if not self._disk().is_dir(container):
            if self._disk().exists(index_path):
                self._disk().remove(index_path)
            return
        rows: dict[str, dict[str, JSONValue]] = {}
        for entry_name in sorted(self._disk().listdir(container)):
            if entry_name in _FORBIDDEN_FILE_NAMES:
                continue
            entry_path = self._disk().join(container, entry_name)
            if not self._disk().is_dir(entry_path):
                continue
            try:
                child = cls.from_disk(entry_path, self)
            except (FileNotFoundError, OSError):
                continue
            if isinstance(child, cls):
                rows[child._name] = child._to_index_row()
        self._disk().atomic_write_json(index_path, rows)

    def remove_folder(self, name: str, *, cls: type[Folder]) -> None:
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            child_dir = cls.child_dir(self, candidate)
            if self._disk().is_dir(child_dir):
                self._disk().remove(child_dir, recursive=True)
                self._children_cache.pop(candidate, None)
                self._remove_index_row(cls, candidate)
                return
            cached = self._children_cache.get(candidate)
            if isinstance(cached, cls):
                self._children_cache.pop(candidate, None)
                self._remove_index_row(cls, candidate)
                return
        raise cls._not_found_error_cls(name)

    # ── Container dir + index helpers ────────────────────────────────────

    @classmethod
    def _container_dir(cls, parent: Folder) -> Path:
        return Path(parent._disk().dirname(cls.child_dir(parent, "_probe_")))

    def _upsert_index_row(self, child: Folder) -> None:
        child_cls = type(child)
        fpath = self._index_path_for(child_cls)
        rows: dict[str, dict[str, JSONValue]] = {}
        if self._disk().exists(fpath):
            try:
                with self._disk().open(fpath) as fh:
                    raw: object = json.load(fh)
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        rows[str(k)] = cast("dict[str, JSONValue]", v)
        rows[child._name] = child._to_index_row()
        self._disk().atomic_write_json(fpath, rows)

    def _remove_index_row(self, cls: type[Folder], slug: str) -> None:
        fpath = self._index_path_for(cls)
        if not self._disk().exists(fpath):
            return
        try:
            with self._disk().open(fpath) as fh:
                raw: object = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        rows = cast("dict[str, JSONValue]", raw)
        if slug not in rows:
            return
        rows.pop(slug)
        self._disk().atomic_write_json(fpath, rows)

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._metadata.model_dump(mode="json"))

    # ── Delete + move ────────────────────────────────────────────────────

    def delete(self) -> None:
        target = self.resolve()
        if self._disk().exists(target):
            self._disk().remove(target, recursive=True)
        if self._parent is not None:
            self._parent._children_cache.pop(self._name, None)

    def move_to(
        self,
        new_parent: Folder,
        *,
        new_name: str | None = None,
    ) -> None:
        # move_to uses OS-level ``shutil.move`` (local paths only). On a
        # remote-backed folder that would silently operate on the wrong (local)
        # path, so refuse it with a clear error instead.
        if not isinstance(self._disk(), LocalFileSystem) or not isinstance(
            new_parent._disk(), LocalFileSystem
        ):
            raise NotImplementedError(
                "move_to is only supported for local-filesystem folders "
                "(it uses OS-level shutil.move); remote-backed folders cannot be moved."
            )
        target_id = self._name if new_name is None else _validate_name_to_id(new_name)
        # Honor the child class's container layout (``runs/run-<id>``,
        # ``projects/<id>``, …) via the same ``child_dir`` hook that mounting
        # uses — a naive ``new_parent.path()/id`` join would strand the moved
        # folder outside its container and hide it from ``list_folders``.
        target_dir = Path(type(self).child_dir(new_parent, target_id))
        if new_parent._disk().exists(target_dir):
            raise FolderMoveCollisionError(str(self.resolve()), str(target_dir))
        src = self.resolve()
        old_parent = self._parent
        # ``shutil.move`` only creates the final path component, so ensure the
        # container dir (``runs/``, ``projects/``, …) exists under the new parent.
        new_parent._disk().mkdir(
            new_parent._disk().dirname(target_dir), parents=True, exist_ok=True
        )
        shutil.move(str(src), str(target_dir))
        if old_parent is not None:
            old_parent._children_cache.pop(self._name, None)
        self._parent = new_parent
        self._root_path = None
        if getattr(self, "_disk_backend", None) is not None:
            del self._disk_backend
        self._name = target_id
        self._metadata = self._metadata.model_copy(
            update={
                "id": target_id,
                "name": new_name if new_name is not None else self._metadata.name,
                "updated_at": datetime.now(),
            }
        )
        new_parent._children_cache[target_id] = self
        self.write_meta()
        # Children indexes are derived; rebuild both endpoints from on-disk truth
        # so a typed ``list_folders(cls=…)`` on either parent reflects the move.
        if old_parent is not None:
            old_parent.sync_folders(cls=type(self))
        new_parent.sync_folders(cls=type(self))


def append_link(
    src: Folder,
    dst: Folder,
    *,
    text: str | None = None,
    role: EdgeRole = DEFAULT_EDGE_ROLE,
) -> None:
    """Append a typed relative markdown link ``src → dst`` to ``src``'s ``index.md``.

    Writes a real markdown link (relative to *src*'s dir) so
    :meth:`Folder.out_edges` resolves it back to *dst* and
    :meth:`Folder.typed_out_edges` recovers *role*. The graph lives in markdown,
    never in ``meta.json``. The *role* is carried in the link's ``[label]``
    channel via :func:`~molexp.workspace.edges.encode_label` — the default role
    encodes to the bare label, so pre-role output stays byte-identical. Appends
    unconditionally; link dedup remains a future enhancement.

    Shared helper: :meth:`Bundle.link` and :meth:`Note.cite` both delegate here,
    so the single source of the markdown-edge format (and the sole role-writing
    chokepoint) lives in this (lower) module that both import from.

    Args:
        src: The Concept the edge originates from.
        dst: The Concept the edge points to.
        text: Optional link label; defaults to *dst*'s name.
        role: The declared :class:`~molexp.workspace.edges.EdgeRole`; defaults to
            :data:`~molexp.workspace.edges.DEFAULT_EDGE_ROLE`.

    Raises:
        ValueError: If *role* is not a known ``EdgeRole`` — validated before any
            write, so an invalid role leaves ``index.md`` untouched.
    """
    validate_role(role)
    rel = os.path.relpath(str(dst.resolve()), str(src.resolve()))
    rel_posix = PurePosixPath(rel).as_posix()
    label = text if text is not None else dst.name
    encoded = encode_label(role, label)
    existing = src.read_index()
    prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
    src.write_index(f"{prefix}- [{encoded}]({rel_posix})\n")


def concept_from_dir(child_dir: PathArg, parent: Folder) -> Folder:
    """Reconstruct *child_dir* as its registered concept subclass via meta.json.

    Reads the OKF ``meta.json`` ``type`` and resolves it through the knowledge
    concept-type registry (unknown/absent → base :class:`Folder`), then
    delegates to that class's ``from_disk``.
    """
    fs = parent._disk()
    marker = _load_concept_marker_dict(fs, child_dir) or {}
    type_str = str(marker.get("type", ""))
    cls = resolve_concept_type(type_str, Folder)
    return cls.from_disk(child_dir, parent)


__all__ = [
    "INDEX_FILENAME",
    "LEGACY_OPS_DIR",
    "META_JSON_FILENAME",
    "OPS_DIR",
    "WORKSPACE_EXPERIMENT_KIND",
    "WORKSPACE_PROJECT_KIND",
    "WORKSPACE_ROOT_KIND",
    "WORKSPACE_RUN_KIND",
    "Folder",
    "LinkScan",
    "append_link",
    "concept_from_dir",
]
