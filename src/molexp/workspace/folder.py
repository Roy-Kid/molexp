"""Unified folder abstraction for workspace storage.

Introduces ``Folder``: a plain Python class providing the contract every
directory under a workspace satisfies — lazy mkdir, atomic JSON, id /
name / kind validation, parent pointer, generic five-verb CRUD
(``add_folder`` / ``get_folder`` / ``has_folder`` / ``list_folders`` /
``remove_folder``), auto-derived per-class index filenames, lifecycle
metadata, and delete / move operations.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar, NamedTuple, TypeVar, cast

from molexp._typing import JSONValue
from molexp.path import Path

from .base import _load_metadata, _reconstruct, _save_metadata
from .errors import FolderMoveCollisionError
from .fs import FileSystem, PathArg
from .fs_local import LocalFileSystem
from .models import FolderMetadata
from .utils import slugify

if TYPE_CHECKING:
    from .catalog import AssetCatalog
    from .workspace import Workspace

F = TypeVar("F", bound="Folder")

WORKSPACE_ROOT_KIND = "workspace.root"
WORKSPACE_PROJECT_KIND = "workspace.project"
WORKSPACE_EXPERIMENT_KIND = "workspace.experiment"
WORKSPACE_RUN_KIND = "workspace.run"

_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)*$")

_METADATA_FILENAME = "metadata.json"
_FORBIDDEN_FILE_NAMES = {".", ".."}
_CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")

# ── OKF: narrative index.md + markdown-link knowledge graph ──────────────────
# The Open Knowledge Format gives every Folder a human-readable narrative
# (``index.md``) whose markdown links ARE the knowledge graph. This is additive
# — it sits alongside the authoritative ``metadata.json`` and never replaces it.
INDEX_FILENAME = "index.md"
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")  # [text](target)


class LinkScan(NamedTuple):
    """Resolved out-links of a Folder's ``index.md``.

    Attributes:
        concepts: Targets resolving to an existing in-tree dir — the
            knowledge-graph out-edges. (Once OKF ``meta.yaml`` lands as the
            unified concept marker, this narrows to true Concept dirs.)
        external: ``http(s)://`` links.
        other: In-tree targets that don't resolve to a dir.
    """

    concepts: list[str]
    external: list[str]
    other: list[str]


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
    """Base class for every workspace-managed directory.

    Carries a ``parent`` pointer, lazy materialization, atomic JSON IO,
    children listing, lifecycle metadata, delete / move operations, and
    the generic ``attach`` / ``create_child`` / ``get_child`` triplet.

    All I/O goes through the :class:`FileSystem` Protocol — when ``fs`` is
    a :class:`RemoteFileSystem`, the folder tree lives on a remote host
    with no code duplication.
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
        self._fs = fs or (parent._fs if parent else LocalFileSystem())
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

    # ── Path resolution ──────────────────────────────────────────────────
    #
    # ``resolve`` and ``path`` return :class:`molexp.Path` — a subclass of
    # :class:`pathlib.PurePosixPath`.  It is a *pure* path: no ``.exists``,
    # ``.read_text`` or other I/O methods, so it cannot accidentally
    # short-circuit to the local filesystem on a remote-backed folder.
    # All I/O must still flow through ``self._fs``.

    def path(self) -> Path:
        """Return the on-disk path; mkdirs if absent (lazy, idempotent)."""
        target = self.resolve()
        self._fs.mkdir(target, parents=True, exist_ok=True)
        return target

    def resolve(self) -> Path:
        """Walk the parent chain without triggering lazy mkdir."""
        if self._parent is None:
            if self._root_path is None:
                raise RuntimeError(
                    f"folder {self._name!r} (kind={self._kind!r}) is unmounted — "
                    "construct via parent.add_folder(child) or pass root_path="
                )
            return Path(self._fs.join(self._root_path, self._name))
        return Path(self._fs.join(self._parent.resolve(), self._name))

    # ── Index filename ───────────────────────────────────────────────────

    @classmethod
    def _index_filename(cls) -> str:
        return _CAMEL_TO_SNAKE.sub("_", cls.__name__).lower() + ".json"

    # ── Atomic JSON IO ───────────────────────────────────────────────────

    def read_json(self, name: str) -> dict[str, JSONValue]:
        _validate_file_name(name)
        fpath = self._fs.join(self.path(), name)
        with self._fs.open(fpath) as fh:
            raw: object = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{fpath} top-level JSON must be an object, got {type(raw).__name__}")
        return cast("dict[str, JSONValue]", raw)

    def write_json(self, name: str, data: object) -> str:
        _validate_file_name(name)
        fpath = self._fs.join(self.path(), name)
        self._fs.atomic_write_json(fpath, data)
        return fpath

    # ── OKF narrative + markdown-link knowledge graph ─────────────────────

    def read_index(self) -> str:
        """Return the OKF ``index.md`` narrative, or ``""`` if absent."""
        fpath = self._fs.join(self.resolve(), INDEX_FILENAME)
        return self._fs.read_text(fpath) if self._fs.exists(fpath) else ""

    def write_index(self, text: str) -> str:
        """Atomically write the OKF ``index.md`` narrative + markdown links."""
        fpath = self._fs.join(self.path(), INDEX_FILENAME)
        self._fs.atomic_write_text(fpath, text)
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
        for target in _MD_LINK.findall(self.read_index()):
            if target.startswith(("http://", "https://")):
                external.append(target)
                continue
            norm = PurePosixPath(os.path.normpath(base / target))
            concept_dir = norm.parent if norm.name == INDEX_FILENAME else norm
            if self._fs.is_dir(str(concept_dir)):
                concepts.append(str(concept_dir))
            else:
                other.append(target)
        return LinkScan(concepts=concepts, external=external, other=other)

    def out_edges(self) -> list[str]:
        """In-tree Folder link targets — the knowledge-graph out-edges."""
        return self.links().concepts

    # ── Lifecycle ────────────────────────────────────────────────────────

    def materialize(self) -> None:
        meta_path = self._fs.join(self.path(), _METADATA_FILENAME)
        self._metadata = self._metadata.model_copy(update={"updated_at": datetime.now()})
        _save_metadata(self._metadata, meta_path, fs=self._fs)

    def save(self) -> None:
        meta_path = self._fs.join(self.path(), _METADATA_FILENAME)
        self._metadata = self._metadata.model_copy(update={"updated_at": datetime.now()})
        _save_metadata(self._metadata, meta_path, fs=self._fs)

    # ── Children ─────────────────────────────────────────────────────────

    def children(self, kind: str | None = None) -> list[Folder]:
        self_path = self.resolve()
        if not self._fs.is_dir(self_path):
            return []
        result: list[Folder] = []
        for entry_name in sorted(self._fs.listdir(self_path)):
            if entry_name in _FORBIDDEN_FILE_NAMES:
                continue
            entry_path = self._fs.join(self_path, entry_name)
            if not self._fs.is_dir(entry_path):
                continue
            meta_file = self._fs.join(entry_path, _METADATA_FILENAME)
            if not self._fs.exists(meta_file):
                continue
            child_meta = _load_metadata(FolderMetadata, meta_file, fs=self._fs)
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
                    "_fs": self._fs,
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
    # the past, see git history for ``_fs``-drop bugs at every level):
    #
    #   * Always inherit ``_fs`` from ``parent`` so children share the
    #     workspace's filesystem (local / remote). Use
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
    # through ``self._fs``.  This unifies the workspace layer (was: ``str``)
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
        """Where a child with *derived_id* lives under *parent*. Override per subclass layout."""
        return Path(parent._fs.join(parent.path(), derived_id))

    @classmethod
    def base_from_disk_attrs(
        cls,
        parent: Folder,
        meta: FolderMetadata,
    ) -> dict[str, object]:
        """Common attrs dict for ``_reconstruct`` — call this from every
        subclass ``from_disk`` to guarantee ``_fs`` + parent link are set."""
        return {
            "_parent": parent,
            "_name": meta.id,
            "_kind": meta.kind,
            "_root_path": None,
            "_metadata": meta,
            "_children_cache": {},
            "_fs": parent._fs,
        }

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Folder:
        """Generic loader: read ``folder.json`` from *child_dir* and reconstruct."""
        meta_file = parent._fs.join(child_dir, _METADATA_FILENAME)
        if not parent._fs.exists(meta_file):
            raise FileNotFoundError(meta_file)
        child_meta = _load_metadata(FolderMetadata, meta_file, fs=parent._fs)
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

    def add_folder(self, child: Folder) -> Folder:
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
            return cached
        child_dir = target_cls.child_dir(self, slug)
        if self._fs.is_dir(child_dir):
            existing = target_cls.from_disk(child_dir, self)
            self._children_cache[slug] = existing
            return existing
        child._parent = self
        child._root_path = None
        child._fs = self._fs
        child.materialize()
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
            if self._fs.is_dir(child_dir):
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
            if cls.child_dir(self, candidate) and self._fs.is_dir(cls.child_dir(self, candidate)):
                return True
        return False

    def list_folders(self, *, cls: type[F] | None = None) -> list[F]:
        self_path = self.resolve()
        if not self._fs.is_dir(self_path):
            return []
        out: list[F] = []
        if cls is None:
            for entry_name in sorted(self._fs.listdir(self_path)):
                if entry_name in _FORBIDDEN_FILE_NAMES:
                    continue
                entry_path = self._fs.join(self_path, entry_name)
                if not self._fs.is_dir(entry_path):
                    continue
                meta_file = self._fs.join(entry_path, _METADATA_FILENAME)
                if not self._fs.exists(meta_file):
                    continue
                child_meta = _load_metadata(FolderMetadata, meta_file, fs=self._fs)
                child = _reconstruct(
                    Folder,
                    {
                        "_parent": self,
                        "_name": child_meta.id,
                        "_kind": child_meta.kind,
                        "_root_path": None,
                        "_metadata": child_meta,
                        "_children_cache": {},
                        "_fs": self._fs,
                    },
                )
                out.append(cast(F, child))
            return out
        index_path = self._fs.join(self_path, cls._index_filename())
        if not self._fs.exists(index_path):
            self.sync_folders(cls=cls)
            if not self._fs.exists(index_path):
                return []
        try:
            with self._fs.open(index_path) as fh:
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
            if not self._fs.is_dir(child_dir):
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
        index_path = self._fs.join(self.resolve(), cls._index_filename())
        if not self._fs.is_dir(container):
            if self._fs.exists(index_path):
                self._fs.remove(index_path)
            return
        rows: dict[str, dict[str, JSONValue]] = {}
        for entry_name in sorted(self._fs.listdir(container)):
            if entry_name in _FORBIDDEN_FILE_NAMES:
                continue
            entry_path = self._fs.join(container, entry_name)
            if not self._fs.is_dir(entry_path):
                continue
            try:
                child = cls.from_disk(entry_path, self)
            except (FileNotFoundError, OSError):
                continue
            if isinstance(child, cls):
                rows[child._name] = child._to_index_row()
        self._fs.atomic_write_json(index_path, rows)

    def remove_folder(self, name: str, *, cls: type[Folder]) -> None:
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            child_dir = cls.child_dir(self, candidate)
            if self._fs.is_dir(child_dir):
                self._fs.remove(child_dir, recursive=True)
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
        return Path(parent._fs.dirname(cls.child_dir(parent, "_probe_")))

    def _upsert_index_row(self, child: Folder) -> None:
        fpath = self._fs.join(self.resolve(), type(child)._index_filename())
        rows: dict[str, dict[str, JSONValue]] = {}
        if self._fs.exists(fpath):
            try:
                with self._fs.open(fpath) as fh:
                    raw: object = json.load(fh)
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        rows[str(k)] = cast("dict[str, JSONValue]", v)
        rows[child._name] = child._to_index_row()
        self._fs.atomic_write_json(fpath, rows)

    def _remove_index_row(self, cls: type[Folder], slug: str) -> None:
        fpath = self._fs.join(self.resolve(), cls._index_filename())
        if not self._fs.exists(fpath):
            return
        try:
            with self._fs.open(fpath) as fh:
                raw: object = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        rows = cast("dict[str, JSONValue]", raw)
        if slug not in rows:
            return
        rows.pop(slug)
        self._fs.atomic_write_json(fpath, rows)

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._metadata.model_dump(mode="json"))

    # ── Catalog upsert dispatch ──────────────────────────────────────────
    #
    # Workspace / Project / Experiment / Run each publish a row into the
    # root workspace catalog on materialize / save.  Only the *dispatch* —
    # resolve the root workspace catalog, then write the kind-specific row —
    # is shared here; each entity supplies its own payload and
    # ``upsert_<kind>`` call through :meth:`_write_catalog_row`.  The four
    # bodies are semantically distinct, so this is dispatch dedup only, not
    # a literal merge.

    def _catalog_upsert(self) -> None:
        """Publish this node's row into the root workspace catalog.

        Resolves the catalog once (walking to the root workspace) and
        delegates the per-kind payload to :meth:`_write_catalog_row`.
        """
        self._write_catalog_row(self._workspace_catalog)

    @property
    def _workspace_catalog(self) -> AssetCatalog:
        """The root workspace's :class:`AssetCatalog`.

        Walks the parent chain to the root node — the only one without a
        parent, always the :class:`Workspace` for the four catalog-bearing
        entities — and returns its ``catalog``.
        """
        node: Folder = self
        while node._parent is not None:
            node = node._parent
        return cast("Workspace", node).catalog

    def _write_catalog_row(self, catalog: AssetCatalog) -> None:
        """Build this node's catalog record and call ``catalog.upsert_<kind>``.

        Per-kind hook with no shared payload — overridden by the four
        catalog-bearing entities (Workspace / Project / Experiment / Run).
        The default raises so a non-entity ``Folder`` (cache, agent
        sessions) that never calls :meth:`_catalog_upsert` cannot silently
        no-op into the catalog.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not participate in the workspace catalog"
        )

    # ── Delete + move ────────────────────────────────────────────────────

    def delete(self) -> None:
        target = self.resolve()
        if self._fs.exists(target):
            self._fs.remove(target, recursive=True)
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
        if not isinstance(self._fs, LocalFileSystem) or not isinstance(
            new_parent._fs, LocalFileSystem
        ):
            raise NotImplementedError(
                "move_to is only supported for local-filesystem folders "
                "(it uses OS-level shutil.move); remote-backed folders cannot be moved."
            )
        target_id = self._name if new_name is None else _validate_name_to_id(new_name)
        target_dir = Path(new_parent._fs.join(new_parent.path(), target_id))
        if new_parent._fs.exists(target_dir):
            raise FolderMoveCollisionError(str(self.resolve()), str(target_dir))
        src = self.resolve()
        dst = target_dir
        shutil.move(str(src), str(dst))
        if self._parent is not None:
            self._parent._children_cache.pop(self._name, None)
        self._parent = new_parent
        self._root_path = None
        self._fs = new_parent._fs
        self._name = target_id
        self._metadata = self._metadata.model_copy(
            update={
                "id": target_id,
                "name": new_name if new_name is not None else self._metadata.name,
                "updated_at": datetime.now(),
            }
        )
        new_parent._children_cache[target_id] = self
        meta_path = new_parent._fs.join(target_dir, _METADATA_FILENAME)
        _save_metadata(self._metadata, meta_path, fs=new_parent._fs)


__all__ = [
    "WORKSPACE_EXPERIMENT_KIND",
    "WORKSPACE_PROJECT_KIND",
    "WORKSPACE_ROOT_KIND",
    "WORKSPACE_RUN_KIND",
    "Folder",
]
