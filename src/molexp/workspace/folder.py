"""Unified folder abstraction for workspace storage.

Introduces ``Folder``: a plain Python class providing the contract every
directory under a workspace satisfies — lazy mkdir, atomic JSON, id /
name / kind validation, parent pointer, children listing, lifecycle
metadata, and delete / move operations.

This sub-spec ships only the base class. Sub-spec 02 will refactor
``Workspace`` / ``Project`` / ``Experiment`` / ``Run`` to inherit from
it; sub-spec 03 introduces the system folders (``SessionFolder`` /
``CacheFolder`` / ``CatalogFolder``); sub-spec 04 adds ``PlanFolder``
in the agent layer. Until those sub-specs land, no existing entity
inherits ``Folder``.

The :data:`_KIND_PATTERN` regex (lowercase ASCII, dot-separated,
segment chars ``[a-z0-9_-]``, no leading dot, no path traversal, no
whitespace) was originally defined in :mod:`molexp.workspace.subsystem`
to validate ``SubsystemStore`` kinds. It moves here because ``Folder``
is the new canonical owner of that grammar; ``subsystem.py``
reverse-imports it so behaviour is preserved.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import ClassVar, cast

from molexp._typing import JSONValue

from .base import _load_metadata, _reconstruct, _save_metadata, atomic_write_json
from .errors import FolderMoveCollisionError
from .models import FolderMetadata
from .utils import slugify

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
        parent: Folder | None,
        name: str,
        kind: str,
        root_path: Path | None = None,
    ) -> None:
        if parent is None and root_path is None:
            raise ValueError("Folder: either parent or root_path must be provided")
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
        """
        if self._parent is None:
            assert self._root_path is not None
            return self._root_path / self._name
        return self._parent._compute_path() / self._name

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

    def attach(
        self,
        name: str,
        *,
        kind: str,
        child_cls: type[Folder] | None = None,
        **child_kwargs: object,
    ) -> Folder:
        """Idempotent child constructor — load existing or create + materialize.

        Returns the existing child (from in-memory cache or on-disk
        directory) if present, otherwise constructs + materializes a
        new one. The derived id (slugified ``name``) is injected as
        the ``id`` kwarg so entity ``__init__`` records the same id
        the cache + child-dir lookup used.

        Args:
            name: Human-readable child name; slugified to form the id.
            kind: Folder kind (``_KIND_PATTERN`` taxonomy); persisted on the child.
            child_cls: Concrete :class:`Folder` subclass to instantiate. Defaults
                to :class:`Folder` itself.
            **child_kwargs: Forwarded to the child constructor on fresh creation.

        Returns:
            The child instance — typed as ``child_cls`` (downcast at the call site).
        """
        target_cls: type[Folder] = Folder if child_cls is None else child_cls
        derived_id = _slugify_name_to_id(name)
        cached = self._children_cache.get(derived_id)
        if cached is not None and isinstance(cached, target_cls):
            return cached
        child_dir = target_cls._child_dir(self, derived_id)
        if child_dir.is_dir():
            child = target_cls._from_disk(child_dir, self)
        else:
            if child_kwargs.get("id") is None:
                child_kwargs["id"] = derived_id
            # Type checkers see ``target_cls`` as ``type[Folder]`` whose
            # ``__init__`` rejects the entity-specific ``**child_kwargs``.
            # The actual entity ``__init__`` (e.g. ``Experiment.__init__``)
            # accepts the kwargs; cast to ``Callable[..., Folder]`` to
            # silence the spurious arity diagnostic.
            ctor = cast("Callable[..., Folder]", target_cls)
            child = ctor(parent=self, name=name, kind=kind, **child_kwargs)
            child.materialize()
        self._children_cache[derived_id] = child
        return child

    def create_child(
        self,
        name: str,
        *,
        kind: str,
        child_cls: type[Folder] | None = None,
        **child_kwargs: object,
    ) -> Folder:
        """Strict child constructor — collision is an error.

        Mirror of :meth:`attach` for callers that must own the create
        moment (CLI / API). The exception class is decided by the typed
        ``child_cls`` so entity factories surface a meaningful
        ``ProjectExistsError`` / ``ExperimentExistsError`` /
        ``RunExistsError`` instead of a generic Folder error.

        Args:
            name: Human-readable child name; slugified to form the id.
            kind: Folder kind (``_KIND_PATTERN`` taxonomy); persisted on the child.
            child_cls: Concrete :class:`Folder` subclass to instantiate.
            **child_kwargs: Forwarded to the child constructor.

        Returns:
            The freshly materialized child.

        Raises:
            Exception: Of type ``child_cls._exists_error_cls`` when a child with
                the derived id already exists in cache or on disk.
        """
        target_cls: type[Folder] = Folder if child_cls is None else child_cls
        derived_id = _slugify_name_to_id(name)
        if derived_id in self._children_cache:
            raise target_cls._exists_error_cls(derived_id)
        if target_cls._child_dir(self, derived_id).is_dir():
            raise target_cls._exists_error_cls(derived_id)
        if child_kwargs.get("id") is None:
            child_kwargs["id"] = derived_id
        ctor = cast("Callable[..., Folder]", target_cls)
        child = ctor(parent=self, name=name, kind=kind, **child_kwargs)
        child.materialize()
        self._children_cache[derived_id] = child
        return child

    def get_child(
        self,
        name_or_id: str,
        *,
        kind: str | None = None,
        child_cls: type[Folder] | None = None,
    ) -> Folder:
        """Strict getter — return the existing child or raise.

        ``kind=None`` skips the kind check (useful when callers only
        have an id). Tries the literal ``name_or_id`` first, then its
        slugified form.

        Args:
            name_or_id: Either the slug-id or a human-readable name.
            kind: If given, the loaded child's ``kind`` must match exactly.
            child_cls: Concrete :class:`Folder` subclass used to reconstruct
                the on-disk row.

        Returns:
            The cached or freshly loaded child.

        Raises:
            Exception: Of type ``child_cls._not_found_error_cls`` when no child
                resolves under either ``name_or_id`` or its slug.
        """
        target_cls: type[Folder] = Folder if child_cls is None else child_cls
        for candidate in (name_or_id, slugify(name_or_id)):
            if not candidate:
                continue
            cached = self._children_cache.get(candidate)
            if cached is not None and (kind is None or cached.kind == kind):
                return cached
            child_dir = target_cls._child_dir(self, candidate)
            if child_dir.is_dir():
                loaded = target_cls._from_disk(child_dir, self)
                if kind is None or loaded.kind == kind:
                    self._children_cache[loaded._name] = loaded
                    return loaded
        raise target_cls._not_found_error_cls(name_or_id)

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
