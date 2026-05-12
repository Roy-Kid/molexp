"""Unified folder abstraction for workspace storage.

Introduces ``Folder``: a plain Python class providing the contract every
directory under a workspace satisfies — lazy mkdir, atomic JSON, id /
name / kind validation, parent pointer, generic five-verb CRUD
(``add_folder`` / ``get_folder`` / ``has_folder`` / ``list_folders`` /
``remove_folder``), auto-derived per-class index filenames, lifecycle
metadata, and delete / move operations.

The :data:`_KIND_PATTERN` regex (lowercase ASCII, dot-separated,
segment chars ``[a-z0-9_-]``, no leading dot, no path traversal, no
whitespace) is the canonical grammar for ``Folder.kind`` and the
slugified-id form of ``Folder.name``.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import ClassVar, TypeVar, cast

from molexp._typing import JSONValue

from .base import _load_metadata, _reconstruct, _save_metadata, atomic_write_json
from .errors import FolderMoveCollisionError
from .models import FolderMetadata
from .utils import slugify

F = TypeVar("F", bound="Folder")

# ── Folder kind taxonomy ─────────────────────────────────────────────────────
#
# Stable identifiers for the four workspace entity subclasses. Grouped
# here (rather than in each entity module) so `grep WORKSPACE_*_KIND`
# returns the whole taxonomy. Sub-spec 03 adds `workspace.session /
# workspace.cache / workspace.catalog`; sub-spec 04 adds `agent.plan`.

WORKSPACE_ROOT_KIND = "workspace.root"
WORKSPACE_PROJECT_KIND = "workspace.project"
WORKSPACE_EXPERIMENT_KIND = "workspace.experiment"
WORKSPACE_RUN_KIND = "workspace.run"

_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)*$")
"""Folder ``kind`` / ``name`` (post-slugify) / ``id`` grammar.

Lowercase ASCII; segments separated by ``.``; segment chars
``[a-z0-9][a-z0-9_-]*``; no leading dot, no path traversal, no
whitespace. Owned by this module; ``subsystem.py`` reverse-imports the
same object for ``SubsystemStore.kind`` validation (until sub-spec 03
deletes that class).
"""

_METADATA_FILENAME = "metadata.json"

_FORBIDDEN_FILE_NAMES = {".", ".."}

# Auto-derived index filename: ``cls.__name__`` → snake_case → ``<name>.json``.
# Used by ``Folder._index_filename()`` so every Folder subclass gets a
# class-named children index file without per-subclass enumeration:
# ``Project`` → ``project.json``; ``AgentSession`` → ``agent_session.json``;
# ``Workspace`` → ``workspace.json`` (overlaps with the workspace's own
# metadata filename, which lives at a different path — no on-disk collision).
_CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def _validate_kind(kind: str) -> None:
    """Raise ``ValueError`` if ``kind`` does not match :data:`_KIND_PATTERN`.

    Public-internal helper: ``subsystem.SubsystemStore`` reuses this for
    its own kind validation so the rule stays single-sourced.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("folder kind must be a non-empty string")
    if not _KIND_PATTERN.fullmatch(kind):
        raise ValueError(
            f"invalid folder kind {kind!r}: must be dotted lowercase ASCII "
            "(e.g. 'workspace.project'); no path separators, leading dots, "
            "uppercase, or whitespace"
        )


def _validate_file_name(name: str) -> None:
    """Reject file names that would escape the folder root."""
    if not isinstance(name, str) or not name:
        raise ValueError("folder file name must be a non-empty string")
    if name in _FORBIDDEN_FILE_NAMES or "/" in name or "\\" in name:
        raise ValueError(f"invalid folder file name {name!r}")


def _validate_name_to_id(name: str) -> str:
    """Validate ``name`` against :data:`_KIND_PATTERN` and return its slug.

    Per spec § Design § Folder ``name``: both the input ``name`` AND the
    slugified ``id`` must match ``_KIND_PATTERN``. The strict pre-check
    rejects uppercase, path traversal (``..``), whitespace, and leading
    dots before slugify silently normalizes them — which would surprise
    callers who passed e.g. ``"UPPER"`` expecting a literal directory.

    Used by direct :class:`Folder` construction (``Folder.__init__``).
    The lenient counterpart :func:`_slugify_name_to_id` is used by
    entity factory wrappers via :meth:`Folder.attach` so user-friendly
    names like ``"My Project"`` flow through.
    """
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


def _slugify_name_to_id(name: str) -> str:
    """Slugify ``name`` and verify the result is a valid id.

    Lenient counterpart to :func:`_validate_name_to_id` for entity
    factory wrappers (``Workspace.Project`` / ``Project.Experiment`` /
    ``Experiment.Run``) that need to accept human names with spaces or
    uppercase. The slug must still be non-empty and kind-pattern-safe;
    only the pre-slugify strictness is dropped.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("folder name must be a non-empty string")
    derived = slugify(name)
    if not derived or not _KIND_PATTERN.fullmatch(derived):
        raise ValueError(f"folder name {name!r} produced invalid id {derived!r}")
    return derived


class Folder:
    """Base class for every workspace-managed directory.

    Carries a ``parent`` pointer, lazy materialization, atomic JSON IO,
    children listing, lifecycle metadata, delete / move operations, and
    the generic ``attach`` / ``create_child`` / ``get_child`` triplet.
    See ``unify-folder-abstraction-01-folder-base`` and
    ``unify-folder-abstraction-02-workspace-subclassing`` specs for the
    contract.

    Construction is side-effect-free: :meth:`path` lazily mkdirs on
    first call. Use :meth:`materialize` to additionally write the
    lifecycle metadata file (``metadata.json``) to disk.

    Plain Python class (not pydantic) because it carries live runtime
    references — ``parent`` and ``_children_cache`` would violate the
    "no ``arbitrary_types_allowed``" project rule (CLAUDE.md
    `## Data type ownership`).
    """

    # ── Subclass-overridable hooks (sub-spec 02) ───────────────────────────
    #
    # Entity subclasses set these to customize the generic ``attach`` /
    # ``create_child`` / ``get_child`` machinery for their on-disk layout
    # and typed exceptions. Default values mean "naked Folder" semantics:
    # child path is ``parent.path() / id``, child metadata is
    # ``metadata.json``, and collisions raise ``ValueError`` /
    # ``LookupError``.

    _exists_error_cls: ClassVar[type[Exception]] = ValueError
    _not_found_error_cls: ClassVar[type[Exception]] = LookupError

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str,
        root_path: Path | None = None,
    ) -> None:
        # parent=None + root_path=None  → unmounted (no path yet; cannot
        # materialize until ``parent.add_folder(self)`` mounts it).
        # parent=Folder                  → mounted under another folder.
        # root_path=Path                 → IS the root (Workspace only).
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
        self._root_path = root_path
        self._metadata = FolderMetadata(id=derived_id, name=name, kind=kind)
        self._children_cache: dict[str, Folder] = {}

    # ── Properties ─────────────────────────────────────────────────────────

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
        """Direct accessor for the :class:`FolderMetadata` lifecycle view.

        Entity subclasses (``Project`` / ``Experiment`` / ``Run``) shadow
        :attr:`metadata` to return their entity-specific pydantic model
        (``ProjectMetadata`` / ``ExperimentMetadata`` / ``RunMetadata``).
        This property is the kind-uniform alternative — guaranteed to
        return the Folder lifecycle view on every subclass.
        """
        return self._metadata

    # ── Path resolution ────────────────────────────────────────────────────

    def path(self) -> Path:
        """Return the on-disk path; mkdirs the directory if absent.

        Lazy: never invoked by ``__init__``. Idempotent: a second call
        is a no-op mkdir (``exist_ok=True``).
        """
        target = self._compute_path()
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _compute_path(self) -> Path:
        """Walk the parent chain without triggering lazy mkdir.

        Used by :meth:`children` and :meth:`delete` so reads on
        non-materialized folders don't side-effect a directory creation.

        Raises:
            RuntimeError: When the folder is in the unmounted state
                (``parent`` and ``root_path`` both ``None``) — i.e. the
                instance was constructed but not yet attached to a parent
                via ``parent.add_folder(self)``.
        """
        if self._parent is None:
            if self._root_path is None:
                raise RuntimeError(
                    f"folder {self._name!r} (kind={self._kind!r}) is unmounted — "
                    "construct via parent.add_folder(child) or pass root_path="
                )
            return self._root_path / self._name
        return self._parent._compute_path() / self._name

    # ── Auto-derived per-class index filename ──────────────────────────────

    @classmethod
    def _index_filename(cls) -> str:
        """Return ``<cls.__name__ as snake_case>.json``.

        The children index of any Folder subclass ``X`` is stored at
        ``<parent_path>/<x_snake>.json``. ``Folder`` → ``folder.json``;
        ``Project`` → ``project.json``; ``Experiment`` → ``experiment.json``;
        ``Run`` → ``run.json``; ``Workspace`` → ``workspace.json``;
        ``AgentSession`` → ``agent_session.json``.

        Same-name collisions with the parent's own metadata file (e.g.
        a Project's own ``project.json`` vs. a children index ``project.json``)
        are avoided by virtue of different paths — the parent's metadata
        sits at ``<parent_path>/<parent_class_snake>.json`` while the
        children index sits at ``<self_path>/<child_class_snake>.json``.

        Constraint: a Folder must not have direct children of its own
        class (would collide self-metadata with child-index filename).
        """
        return _CAMEL_TO_SNAKE.sub("_", cls.__name__).lower() + ".json"

    # ── Atomic JSON IO ─────────────────────────────────────────────────────

    def read_json(self, name: str) -> dict[str, JSONValue]:
        """Read a JSON object stored under this folder.

        Raises ``FileNotFoundError`` (stdlib pass-through) when the file
        is absent. Raises ``ValueError`` when the file's top-level JSON
        is not an object (deserialization-boundary narrowing per
        CLAUDE.md type-safety rule).
        """
        _validate_file_name(name)
        path = self.path() / name
        with path.open() as fh:
            raw: object = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path} top-level JSON must be an object, got {type(raw).__name__}")
        return cast("dict[str, JSONValue]", raw)

    def write_json(self, name: str, data: object) -> Path:
        """Write a JSON payload atomically; returns the destination path."""
        _validate_file_name(name)
        path = self.path() / name
        atomic_write_json(path, data)
        return path

    # ── Lifecycle (metadata persistence) ───────────────────────────────────

    def materialize(self) -> None:
        """Create the folder directory and persist metadata.

        Bumps ``updated_at = datetime.now()`` so a subsequent
        :meth:`save` produces a strictly greater timestamp.
        """
        meta_path = self.path() / _METADATA_FILENAME
        self._metadata = self._metadata.model_copy(update={"updated_at": datetime.now()})
        _save_metadata(self._metadata, meta_path)

    def save(self) -> None:
        """Persist a fresh ``updated_at`` plus any in-memory metadata changes."""
        meta_path = self.path() / _METADATA_FILENAME
        self._metadata = self._metadata.model_copy(update={"updated_at": datetime.now()})
        _save_metadata(self._metadata, meta_path)

    # ── Children ───────────────────────────────────────────────────────────

    def children(self, kind: str | None = None) -> list[Folder]:
        """Scan the on-disk dir for child folders, optionally filtered by ``kind``.

        Filtering is on the persisted ``FolderMetadata.kind`` field of
        each child — Python ``type`` is deliberately not consulted so
        Folder subclasses across the chain don't need forward references.

        Returns ``[]`` when ``self`` has not been materialized yet,
        preserving the lazy semantic (no side-effecting mkdir on a
        read-shaped query).
        """
        self_path = self._compute_path()
        if not self_path.is_dir():
            return []
        result: list[Folder] = []
        for entry in sorted(self_path.iterdir()):
            if not entry.is_dir():
                continue
            meta_file = entry / _METADATA_FILENAME
            if not meta_file.exists():
                continue
            child_meta = _load_metadata(FolderMetadata, meta_file)
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

    # ── Generic attach / create_child / get_child (sub-spec 02) ────────────

    @classmethod
    def _child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Return the on-disk dir for a child of *cls* under ``parent``.

        Default: ``parent.path() / derived_id``. Entity subclasses
        override this to inject the namespace dir (``"projects"`` /
        ``"experiments"`` / ``"runs"``) and any prefix the on-disk
        layout requires (e.g. ``Run`` uses ``"run-<id>"``).
        """
        return parent.path() / derived_id

    @classmethod
    def _from_disk(cls, child_dir: Path, parent: Folder) -> Folder:
        """Reconstruct an instance of *cls* from an existing on-disk dir.

        Default: read ``metadata.json`` and rebuild a vanilla
        :class:`Folder`. Entity subclasses (which persist to
        ``<entity>.json`` instead of ``metadata.json`` and need entity-
        specific state restored) override this hook.
        """
        meta_file = child_dir / _METADATA_FILENAME
        if not meta_file.exists():
            raise FileNotFoundError(meta_file)
        child_meta = _load_metadata(FolderMetadata, meta_file)
        return _reconstruct(
            cls,
            {
                "_parent": parent,
                "_name": child_meta.id,
                "_kind": child_meta.kind,
                "_root_path": None,
                "_metadata": child_meta,
                "_children_cache": {},
            },
        )

    # ── Generic five-verb CRUD ─────────────────────────────────────────────
    #
    # ``add_folder / get_folder / has_folder / list_folders / remove_folder``
    # operate on any ``Folder`` subclass via the ``cls=`` kwarg. Typed
    # subclasses layer one-line ``add_<noun> / get_<noun> / has_<noun> /
    # list_<noun>s / remove_<noun>`` sugar on top.

    def add_folder(self, child: Folder) -> Folder:
        """Mount a pre-constructed unmounted ``Folder`` as a child of ``self``.

        - Wires ``child._parent = self``.
        - Materializes ``child`` on disk (its own ``<cls>.json`` metadata).
        - Appends a row to ``self.path() / child.__class__._index_filename()``.
        - Caches ``child`` in ``self._children_cache``.

        **Idempotent on (slug, kind)**: if a child with the same slugified
        name and kind already exists (in-memory cache or on-disk dir),
        returns the existing one — the passed-in ``child`` is dropped.

        Args:
            child: A freshly-constructed ``Folder`` subclass instance whose
                ``_parent`` is ``None`` (unmounted state). Re-mounting an
                already-mounted folder raises ``ValueError``.

        Returns:
            The mounted child — same instance as ``child`` on fresh mount,
            or the cached / on-disk instance on idempotent collision.
        """
        if child._parent is not None or child._root_path is not None:
            raise ValueError(
                f"folder {child._name!r} (kind={child._kind!r}) is already mounted; "
                "add_folder() accepts only unmounted folders"
            )
        target_cls = type(child)
        slug = child._name  # already slugified by __init__
        # Idempotent: check cache first, then on-disk.
        cached = self._children_cache.get(slug)
        if cached is not None and cached._kind == child._kind:
            return cached
        child_dir = target_cls._child_dir(self, slug)
        if child_dir.is_dir():
            existing = target_cls._from_disk(child_dir, self)
            self._children_cache[slug] = existing
            return existing
        # Fresh mount.
        child._parent = self
        child._root_path = None
        child.materialize()
        self._children_cache[slug] = child
        self._upsert_index_row(child)
        return child

    def get_folder(self, name: str, *, cls: type[F]) -> F:
        """Strict getter — return the existing child or raise.

        Tries the literal ``name`` first, then ``slugify(name)``. ``cls``
        decides both the kind filter (via ``cls`` instance check) and the
        typed reconstruction (via ``cls._from_disk``).

        Args:
            name: Either the slug-id or the human-readable name.
            cls: Concrete ``Folder`` subclass to filter by + reconstruct as.

        Raises:
            Exception: Of type ``cls._not_found_error_cls`` when no child
                resolves under either ``name`` or its slug as an instance
                of ``cls``.
        """
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            cached = self._children_cache.get(candidate)
            if isinstance(cached, cls):
                return cached
            child_dir = cls._child_dir(self, candidate)
            if child_dir.is_dir():
                loaded = cls._from_disk(child_dir, self)
                if isinstance(loaded, cls):
                    self._children_cache[loaded._name] = loaded
                    return cast(F, loaded)
        raise cls._not_found_error_cls(name)

    def has_folder(self, name: str, *, cls: type[Folder]) -> bool:
        """Existence check — never raises, never materializes anything."""
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            cached = self._children_cache.get(candidate)
            if isinstance(cached, cls):
                return True
            if cls._child_dir(self, candidate).is_dir():
                return True
        return False

    def list_folders(self, *, cls: type[F] | None = None) -> list[F]:
        """List children, optionally filtered by class.

        Reads the per-class index file (``<self_path>/<cls_snake>.json``)
        as the **authoritative source of truth**. If the index is missing
        on first access, performs an opportunistic :meth:`sync_folders`
        to bring it in line with the on-disk directory contents — so
        the first call after a fresh checkout never returns spurious
        empties.

        Without ``cls``: returns all ``Folder`` children discovered on
        disk (any class — falls back to vanilla ``Folder`` reconstruction).

        Index ↔ directory drift is **not** a normal state. Use
        :meth:`sync_folders` when external tooling has touched the
        on-disk layout out-of-band; otherwise ``add_folder`` /
        ``remove_folder`` keep the two in sync.
        """
        self_path = self._compute_path()
        if not self_path.is_dir():
            return []
        out: list[F] = []
        if cls is None:
            # Vanilla discovery: scan immediate subdirs for any metadata.json.
            for entry in sorted(self_path.iterdir()):
                if not entry.is_dir():
                    continue
                meta_file = entry / _METADATA_FILENAME
                if not meta_file.exists():
                    continue
                child_meta = _load_metadata(FolderMetadata, meta_file)
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
        # Typed discovery: read the per-class index file.
        index_path = self_path / cls._index_filename()
        if not index_path.exists():
            # Opportunistic sync: rebuild the index from disk so subsequent
            # reads are O(1) per row. Skips when the container dir is empty.
            self.sync_folders(cls=cls)
            if not index_path.exists():
                return []
        try:
            with index_path.open() as fh:
                raw: object = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, dict):
            return []
        for slug in raw:
            cached = self._children_cache.get(str(slug))
            if isinstance(cached, cls):
                out.append(cast(F, cached))
                continue
            child_dir = cls._child_dir(self, str(slug))
            if not child_dir.is_dir():
                # Stale row — index drifted from disk. Skip; caller can
                # ``sync_folders`` to clean up.
                continue
            try:
                loaded = cls._from_disk(child_dir, self)
            except (FileNotFoundError, OSError):
                continue
            if isinstance(loaded, cls):
                self._children_cache[loaded._name] = loaded
                out.append(cast(F, loaded))
        return out

    def sync_folders(self, *, cls: type[Folder]) -> None:
        """Reconcile the per-class index file with on-disk reality.

        Walks ``<self_path>/<cls._container_dir>``, identifies every
        directory that reconstructs cleanly as an instance of ``cls``,
        and atomically rewrites ``<self_path>/<cls_snake>.json`` so its
        rows exactly match what is on disk. Adds rows for new
        directories, drops rows for vanished ones.

        Called automatically by :meth:`list_folders` on first read when
        the index file is missing. Use explicitly when external tooling
        (rsync, manual rm, legacy migration) has touched the directory
        out-of-band.
        """
        container = cls._container_dir(self)
        index_path = self._compute_path() / cls._index_filename()
        if not container.is_dir():
            if index_path.exists():
                index_path.unlink()
            return
        rows: dict[str, dict[str, JSONValue]] = {}
        for entry in sorted(container.iterdir()):
            if not entry.is_dir():
                continue
            try:
                child = cls._from_disk(entry, self)
            except (FileNotFoundError, OSError):
                continue
            if isinstance(child, cls):
                rows[child._name] = child._to_index_row()
        atomic_write_json(index_path, rows)

    def remove_folder(self, name: str, *, cls: type[Folder]) -> None:
        """Delete child dir + drop its row from the index file + drop cache."""
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            child_dir = cls._child_dir(self, candidate)
            if child_dir.is_dir():
                shutil.rmtree(child_dir)
                self._children_cache.pop(candidate, None)
                self._remove_index_row(cls, candidate)
                return
            cached = self._children_cache.get(candidate)
            if isinstance(cached, cls):
                self._children_cache.pop(candidate, None)
                self._remove_index_row(cls, candidate)
                return
        raise cls._not_found_error_cls(name)

    # ── Per-class container dir + index helpers ────────────────────────────

    @classmethod
    def _container_dir(cls, parent: Folder) -> Path:
        """Return the directory holding sibling-instances of ``cls`` under ``parent``.

        Default: ``parent.path() / <cls.__name__ snake_case>s`` plural —
        e.g. ``projects/``, ``experiments/``, ``runs/``. Typed subclasses
        that already override ``_child_dir`` derive their container from
        that hook automatically (one entry up). Subclasses with a custom
        layout (e.g. ``Run`` uses ``runs/run-<id>/``) can override this
        directly.
        """
        sample = cls._child_dir(parent, "_probe_")
        # Strip the slug component; the parent of any individual child
        # dir IS the container dir.
        return sample.parent

    def _upsert_index_row(self, child: Folder) -> None:
        """Append/refresh ``child``'s row in this folder's children index.

        Row schema: derived from the child's persisted ``FolderMetadata``
        (id / name / kind / created_at / updated_at). Subclasses that want
        richer index columns override ``_index_row_for(child)``.

        Atomic: writes through ``atomic_write_json``.
        """
        path = self._compute_path() / type(child)._index_filename()
        rows: dict[str, dict[str, JSONValue]] = {}
        if path.exists():
            try:
                with path.open() as fh:
                    raw: object = json.load(fh)
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        rows[str(k)] = cast("dict[str, JSONValue]", v)
        rows[child._name] = child._to_index_row()
        atomic_write_json(path, rows)

    def _remove_index_row(self, cls: type[Folder], slug: str) -> None:
        path = self._compute_path() / cls._index_filename()
        if not path.exists():
            return
        try:
            with path.open() as fh:
                raw: object = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        if slug not in raw:
            return
        raw.pop(slug)
        atomic_write_json(path, raw)

    def _to_index_row(self) -> dict[str, JSONValue]:
        """Return the schema this Folder uses when its parent indexes it.

        Default: project ``FolderMetadata`` via ``model_dump(mode='json')``.
        Entity subclasses (``Project`` / ``Experiment`` / ``Run`` /
        ``Agent`` / ``AgentSession`` / ``PlanFolder``) override to expose
        richer columns from their entity-specific metadata.
        """
        return cast("dict[str, JSONValue]", self._metadata.model_dump(mode="json"))

    # ── Delete + move ──────────────────────────────────────────────────────

    def delete(self) -> None:
        """Remove the folder tree on disk and drop self from parent's cache.

        Safe to call after a sibling already removed the path — the
        rmtree is guarded by ``target.exists()`` so the operation stays
        idempotent. Path resolution avoids :meth:`path` so we don't
        re-mkdir the target as a side-effect.
        """
        target = self._compute_path()
        if target.exists():
            shutil.rmtree(target)
        if self._parent is not None:
            self._parent._children_cache.pop(self._name, None)

    def move_to(
        self,
        new_parent: Folder,
        *,
        new_name: str | None = None,
    ) -> None:
        """Move the folder tree under ``new_parent``, optionally renaming.

        Raises :class:`FolderMoveCollisionError` if the destination
        path already exists. On success: parent pointer is rewired,
        name updated (when ``new_name`` provided), ``updated_at`` is
        bumped, and the metadata file under the new location is
        rewritten so a subsequent reload reflects the move.
        """
        target_id = self._name if new_name is None else _validate_name_to_id(new_name)
        target_dir = new_parent.path() / target_id
        if target_dir.exists():
            raise FolderMoveCollisionError(str(self._compute_path()), str(target_dir))
        shutil.move(str(self._compute_path()), str(target_dir))
        if self._parent is not None:
            self._parent._children_cache.pop(self._name, None)
        self._parent = new_parent
        self._root_path = None
        self._name = target_id
        self._metadata = self._metadata.model_copy(
            update={
                "id": target_id,
                "name": new_name if new_name is not None else self._metadata.name,
                "updated_at": datetime.now(),
            }
        )
        new_parent._children_cache[target_id] = self
        _save_metadata(self._metadata, target_dir / _METADATA_FILENAME)


__all__ = [
    "WORKSPACE_EXPERIMENT_KIND",
    "WORKSPACE_PROJECT_KIND",
    "WORKSPACE_ROOT_KIND",
    "WORKSPACE_RUN_KIND",
    "Folder",
]
