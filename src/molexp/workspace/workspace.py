"""Workspace: top-level container with project management.

The Workspace is the root of the hierarchy and the only :class:`Folder`
whose ``parent`` is ``None``. Unlike lower levels, the workspace
constructor **does** ensure its root directory + ``workspace.json``
exist, because every other level needs a materialized workspace as
anchor.

Child factories (``.add_project(...)``) are idempotent: they load existing
children from disk or create + materialize new ones.
"""

from __future__ import annotations

from pathlib import Path as _LocalPath
from typing import cast

from molexp.knowledge.types import concept_type
from molexp.path import Path

from .assets import AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _load_metadata,
    _save_metadata,
)
from .cache import CacheFolder
from .catalog import AssetCatalog
from .errors import ProjectExistsError, ProjectNotFoundError
from .folder import (
    WORKSPACE_PROJECT_KIND,
    WORKSPACE_ROOT_KIND,
    Folder,
)
from .fs import FileSystem, PathArg
from .fs_local import LocalFileSystem
from .library import Library
from .models import FolderMetadata, WorkspaceMetadata
from .project import Project
from .utils import slugify

# CLI-level root override: set by ``molexp run`` before executing the user
# script so a script's ``me.Workspace(...)`` resolves against the CLI instead
# of (or, when rootless, in place of) its hardcoded argument. Stored as
# ``(path, explicit)`` or ``None``:
#   * ``explicit=True``  — an explicit ``-ws/--workspace`` flag: STRONG, wins
#     even over a root the script passed (the original CLI-flag behavior).
#   * ``explicit=False`` — an inferred root (the entry-script directory, set
#     when no flag is given): WEAK, only fills in when the script omits its
#     root, so a script that passes an explicit root keeps it.
# ``None`` means no override — ``Workspace(root)`` uses the caller's root as-is.
_cli_root_override: tuple[_LocalPath, bool] | None = None


def set_cli_root_override(path: _LocalPath | str | None, *, explicit: bool = True) -> None:
    """Set (or clear) the CLI-level workspace root override.

    Args:
        path: The override root, or ``None`` to clear it.
        explicit: ``True`` (default) for an explicit ``-ws`` flag — wins over a
            root the script hardcodes. ``False`` for an inferred root (entry
            script directory) — used only when the script omits its root.

    Intended solely for ``molexp run`` to make the CLI flag authoritative, or
    to fill in a rootless ``Workspace(name=...)`` with the script's directory.
    """
    global _cli_root_override
    _cli_root_override = (_LocalPath(path).resolve(), explicit) if path is not None else None


@concept_type(WORKSPACE_ROOT_KIND)
class Workspace(Folder):
    """Top-level workspace with project management and global asset library.

    Inherits :class:`Folder` (sub-spec 02): ``kind`` is
    :data:`WORKSPACE_ROOT_KIND`, ``parent`` is ``None``. The workspace
    is its own root — :meth:`resolve` returns :attr:`root`
    directly rather than nesting one level deeper like other Folder
    subclasses.

    Example::

        ws = Workspace("./lab")
        project = ws.add_project("QM9")
        exp = project.add_experiment("baseline", params={"lr": 1e-3}, n_replicas=3)
    """

    _exists_error_cls = ProjectExistsError
    _not_found_error_cls = ProjectNotFoundError

    def __init__(
        self, root: PathArg | None = None, name: str | None = None, *, fs: FileSystem | None = None
    ) -> None:
        self._fs = fs or LocalFileSystem()

        # Root precedence (local only). An explicit ``-ws`` override is STRONG
        # and wins over a script-passed root; an inferred override is WEAK and
        # only fills in when ``root`` is omitted. With no override, the passed
        # root is used as-is; with neither root nor override we fail fast.
        local = isinstance(self._fs, LocalFileSystem)
        override = _cli_root_override if local else None
        if override is not None and (root is None or override[1]):
            resolved_raw = str(override[0])
        elif root is None:
            raise ValueError(
                "Workspace root not given and no CLI override set — "
                "pass a root or run the script via `molexp run`"
            )
        elif local:
            resolved_raw = self._fs.resolve(str(root))
        else:
            resolved_raw = str(root)  # Remote: use path as-is (tilde handled by remote shell)

        metadata_path = self._fs.join(resolved_raw, "workspace.json")
        if self._fs.exists(metadata_path):
            entity_meta = _load_metadata(WorkspaceMetadata, metadata_path, fs=self._fs)
        else:
            display_name = name if name is not None else self._fs.basename(resolved_raw)
            entity_meta = WorkspaceMetadata(id=slugify(display_name), name=display_name)

        # Workspace bypasses ``Folder.__init__`` because the human-readable
        # ``name`` may contain characters (e.g. spaces, uppercase) that the
        # ``_KIND_PATTERN`` validator rejects — the slugified ``id`` is the
        # kind-safe form persisted in :class:`FolderMetadata`.
        self._parent = None
        self._name = entity_meta.id
        self._kind = WORKSPACE_ROOT_KIND
        self._root_path: Path = Path(resolved_raw)
        self._metadata = FolderMetadata(
            id=entity_meta.id,
            name=entity_meta.name,
            kind=WORKSPACE_ROOT_KIND,
            created_at=entity_meta.created_at,
            updated_at=entity_meta.created_at,
        )
        self._children_cache = {}

        # Entity-specific state — ``root`` is :class:`molexp.Path` for both
        # local and remote workspaces; wrap with :class:`pathlib.Path` at
        # genuine-local-I/O sites.
        self.root: Path = self._root_path
        self._entity_metadata: WorkspaceMetadata = entity_meta
        self._data_assets: DataAssetLibrary | None = None
        self._library: Library | None = None
        self._catalog: AssetCatalog | None = None
        self._cache_folder: CacheFolder | None = None

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> Path:
        """Workspace IS its own on-disk dir; no parent nesting."""
        return self._root_path

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Workspace:
        """Reconstruct a Workspace rooted at *child_dir* (OKF concept rebuild).

        A Workspace is its own root and persists ``workspace.json`` (not the
        base ``metadata.json``), so the generic :meth:`Folder.from_disk` cannot
        rebuild it. ``concept_from_dir`` reaches here when a ``meta.yaml`` typed
        ``workspace.root`` is found; the constructor reloads ``workspace.json``
        from *child_dir* and ignores the synthetic *parent* (a Workspace has
        none). See the Folder.from_disk hook docs.
        """
        return cls(root=child_dir, fs=parent._fs)

    def _ensure_materialized(self) -> None:
        meta_path = self._fs.join(self.resolve(), "workspace.json")
        if not self._fs.exists(meta_path):
            self.materialize()

    # ── Properties (entity-specific) ─────────────────────────────────────

    @property
    def metadata(self) -> WorkspaceMetadata:  # type: ignore[override]
        """Workspace-entity metadata (shadows :attr:`Folder.metadata`).

        :attr:`Folder.folder_metadata` still returns the kind-uniform
        :class:`FolderMetadata` view.
        """
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: WorkspaceMetadata) -> None:
        self._entity_metadata = value

    @property
    def id(self) -> str:
        return self._entity_metadata.id

    @property
    def name(self) -> str:
        return self._entity_metadata.name

    @property
    def created_at(self):  # noqa: ANN201
        return self._entity_metadata.created_at

    @property
    def scope(self) -> AssetScope:
        return AssetScope(kind="workspace", ids=())

    @property
    def assets(self) -> AssetsView:
        """Scope-filtered catalog view (read-only queries)."""
        return AssetsView(self.catalog, self.scope)

    @property
    def data_assets(self) -> DataAssetLibrary:
        """Library for importing ``DataAsset`` inputs."""
        if self._data_assets is None:
            self._data_assets = DataAssetLibrary(self.root, self.scope, self.catalog)
        return self._data_assets

    @property
    def library(self) -> Library:
        """Notes + references store for the workspace scope."""
        if self._library is None:
            self._library = Library(self.root, self.scope, self.catalog)
        return self._library

    @property
    def catalog(self) -> AssetCatalog:
        """Workspace-wide JSON asset + entity catalog.

        Lazily constructed.  The catalog directory is created on first write.
        """
        if self._catalog is None:
            self._catalog = AssetCatalog(self.root)
        return self._catalog

    # ── System folder accessors (singletons via lowercase property) ──────

    @property
    def cache(self) -> CacheFolder:  # type: ignore[override]
        """The (single) :class:`CacheFolder` for this workspace.

        Lazily constructed; identity-stable. ``ws.cache is ws.cache``.
        The underlying ``<root>/cache/`` directory is created on first
        read/write through the folder.
        """
        if self._cache_folder is None:
            self._cache_folder = CacheFolder(
                parent=self,
                name="cache",
                kind="workspace.cache",
            )
        return self._cache_folder

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        root_str = self.resolve()
        self._fs.mkdir(root_str, parents=True, exist_ok=True)
        meta_path = self._fs.join(root_str, "workspace.json")
        _save_metadata(self._entity_metadata, meta_path, fs=self._fs)
        self.write_meta()  # OKF marker for the root, additive
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        meta_path = self._fs.join(self.resolve(), "workspace.json")
        _save_metadata(self._entity_metadata, meta_path, fs=self._fs)
        self._catalog_upsert()

    def _write_catalog_row(self, catalog: AssetCatalog) -> None:
        meta = self._entity_metadata
        catalog.upsert_workspace(
            {
                "workspace_id": meta.id,
                "root_path": self.resolve(),
                "name": meta.name,
                "created_at": meta.created_at.isoformat(),
                "updated_at": meta.created_at.isoformat(),
            }
        )

    # ── Alternative constructors ─────────────────────────────────────────

    @classmethod
    def load(cls, root: PathArg, *, fs: FileSystem | None = None) -> Workspace:
        """Load an existing workspace from disk (raises if missing)."""
        _fs = fs or LocalFileSystem()
        if isinstance(_fs, LocalFileSystem):
            root_path = _LocalPath(root).resolve()
            metadata_file = root_path / "workspace.json"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Workspace metadata not found at {metadata_file}")
            return cls(root, fs=fs)
        # RemoteFileSystem path
        root_str = str(root)
        metadata_path = _fs.join(root_str, "workspace.json")
        if not _fs.exists(metadata_path):
            raise FileNotFoundError(f"Workspace metadata not found at {metadata_path}")
        return cls(root, fs=fs)

    # ── Project CRUD: typed semantic sugar over generic Folder CRUD ────────

    def add_project(self, name: str) -> Project:
        """Mount a project under this workspace (idempotent on slug)."""
        self._ensure_materialized()
        child = self._construct_child(Project, name, fs=self._fs)
        return cast(Project, self.add_folder(child))

    def project(self, name: str) -> Project:
        """Fluent create-or-get alias for :meth:`add_project` (idempotent)."""
        return self.add_project(name)

    def get_project(self, name: str) -> Project:
        """Strict getter — raise :class:`ProjectNotFoundError` if absent."""
        return self.get_folder(name, cls=Project)

    def has_project(self, name: str) -> bool:
        return self.has_folder(name, cls=Project)

    # ``list_projects`` is defined below as the legacy method; semantics
    # match the generic ``list_folders(cls=Project)`` view, so we keep
    # the legacy implementation for now to avoid behavioral drift.

    def remove_project(self, name: str) -> None:
        """Delete project directory + cascade-drop catalog rows + drop indices."""
        slug = slugify(name)
        self.remove_folder(name, cls=Project)
        self.catalog.remove_project(slug)

    def list_projects(self) -> list[Project]:
        """List all projects in this workspace via the typed CRUD view."""
        return self.list_folders(cls=Project)

    def children(self, kind: str | None = None) -> list[Folder]:
        """List entity children (currently only :class:`Project`)."""
        if kind is not None and kind != WORKSPACE_PROJECT_KIND:
            return []
        return list(self.list_projects())
